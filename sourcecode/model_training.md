---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Training Model for annotation

Resources:

* https://www.youtube.com/watch?v=IfsjMg4fLWQ&list=PLtmWHNX-gukKocXQOkQjuVxglSDYWsSh9&index=8&t=0s
* https://towardsdatascience.com/tuned-version-of-seq2seq-tutorial-ddb64db46e2a
* https://github.com/fastai/course-nlp/blob/master/7-seq2seq-translation.ipynb
* https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213
* https://github.com/fastai/fastai/blob/master/courses/dl2/translate.ipynb
* https://github.com/fastai/course-v3/blob/master/nbs/dl2/translation.ipynb
* https://github.com/fastai/course-v3/blob/master/nbs/dl2/translation_transformer.ipynb
* https://stackoverflow.com/questions/38287772/cbow-v-s-skip-gram-why-invert-context-and-target-words


```python
from fastai.text import *
from seq2seq import DataBunch, Seq2SeqTextList, TeacherForcing
import pandas as pd

# in develoment library, see https://github.com/facebookresearch/fastText to download, how to build, etc.
import fasttext as ft
```

```python
# to help with getting consistent results
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

# at batch_size 64, uses up to 50% of GPU.
#                   uses about 7GGB of GPU RAM
#                   epochs are about 35 sec each 
# at batch_size 128, uses up to 75% of GPU
#                    uses about 16GB of GPU RAM
#                    epochs are about 20 sec each
# while larger batch size trains faster, it appears to have lower overall accuracy from the same number of epochs
batch_size = 64
```

```python
path = Path('../data')
in_data_file = path/'processed_data.csv'
databunch_file = path/'databunch.pkl'
```

```python
# Uncomment this to download the pre-trained vectors for fasttext
# ! wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz -P {path}
# ! gunzip {path}/cc.en.300.bin.gz
```

```python
df = pd.read_csv(in_data_file, low_memory=False, na_filter=False)
df.head()
```

```python
df.tail()
```

```python
src = (Seq2SeqTextList.from_df(df, path = path, cols='annotated_text')
       .split_by_rand_pct(seed=seed)
       .label_from_df(cols='orig_text', label_cls=TextList))
```

```python
len(src.train) + len(src.valid)
```

```python
data = src.databunch(bs=batch_size)
```

```python
data.save(databunch_file)
```

```python
data.show_batch()
```

```python
# this takes a minute or two to load
en_vecs = ft.load_model(str((path/'cc.en.300.bin')))
```

```python
# check a few words to make sure they work
# see https://fasttext.cc/docs/en/unsupervised-tutorial.html
# model includes character n-grams, so unknown words will typically have vectors based on contained substrings
for w in ['diabetic', 'sgot', 'thrombectomy', 'revascularization']:
    print(en_vecs.get_word_vector(w))
```

```python
def create_emb(vecs, itos, em_sz=300, mult=1.):
    emb = nn.Embedding(len(itos), em_sz, padding_idx=1)
    wgts = emb.weight.data
    vec_dic = {w:vecs.get_word_vector(w) for w in vecs.get_words()}
    miss = []
    for i,w in enumerate(itos):
        try: wgts[i] = tensor(vec_dic[w])
        except: miss.append(w)
    return emb
```

```python
len(data.x.vocab.itos)
```

```python
len(data.y.vocab.itos)
```

```python
len(data.vocab.stoi)
```

```python
len(data.vocab.itos)
```

```python
emb_enc = None
emb_dec = None
if not os.path.exists(path/'emb_enc.pth'):
    # creating embedding takes several minutes 5? maybe more
    print('Creating embedding')
    emb_enc = create_emb(en_vecs, data.x.vocab.itos)
    emb_dec = create_emb(en_vecs, data.y.vocab.itos)
    del en_vecs # release memory from vector
    torch.save(emb_enc, path/'emb_enc.pth')
    torch.save(emb_dec, path/'emb_dec.pth')
else:
    print('Loading embedding')
    emb_enc = torch.load(path/'emb_enc.pth')
    emb_dec = torch.load(path/'emb_dec.pth')
```

```python
emb_enc.weight.size(), emb_dec.weight.size()
```

```python
class Seq2SeqRNN(nn.Module):
    def __init__(self, emb_enc, emb_dec, 
                    nh, out_sl, 
                    nl=2, bos_idx=0, pad_idx=1):
        super().__init__()
        self.nl,self.nh,self.out_sl = nl,nh,out_sl
        self.bos_idx,self.pad_idx = bos_idx,pad_idx
        self.em_sz_enc = emb_enc.embedding_dim
        self.em_sz_dec = emb_dec.embedding_dim
        self.voc_sz_dec = emb_dec.num_embeddings
                 
        self.emb_enc = emb_enc
        self.emb_enc_drop = nn.Dropout(0.15)
        self.gru_enc = nn.GRU(self.em_sz_enc, nh, num_layers=nl,
                              dropout=0.25, batch_first=True)
        self.out_enc = nn.Linear(nh, self.em_sz_dec, bias=False)
        
        self.emb_dec = emb_dec
        self.gru_dec = nn.GRU(self.em_sz_dec, self.em_sz_dec, num_layers=nl,
                              dropout=0.1, batch_first=True)
        self.out_drop = nn.Dropout(0.35)
        self.out = nn.Linear(self.em_sz_dec, self.voc_sz_dec)
        self.out.weight.data = self.emb_dec.weight.data
        
    def encoder(self, bs, inp):
        h = self.initHidden(bs)
        emb = self.emb_enc_drop(self.emb_enc(inp))
        _, h = self.gru_enc(emb, h)
        h = self.out_enc(h)
        return h
    
    def decoder(self, dec_inp, h):
        emb = self.emb_dec(dec_inp).unsqueeze(1)
        outp, h = self.gru_dec(emb, h)
        outp = self.out(self.out_drop(outp[:,0]))
        return h, outp
        
    def forward(self, inp):
        bs, sl = inp.size()
        h = self.encoder(bs, inp)
        dec_inp = inp.new_zeros(bs).long() + self.bos_idx
        
        res = []
        for i in range(self.out_sl):
            h, outp = self.decoder(dec_inp, h)
            dec_inp = outp.max(1)[1]
            res.append(outp)
            if (dec_inp==self.pad_idx).all(): break
        return torch.stack(res, dim=1)
    
    def initHidden(self, bs): return one_param(self).new_zeros(self.nl, bs, self.nh)
```

```python
def seq2seq_loss(out, targ, pad_idx=1):
    bs,targ_len = targ.size()
    _,out_len,vs = out.size()
    if targ_len>out_len: out  = F.pad(out,  (0,0,0,targ_len-out_len,0,0), value=pad_idx)
    if out_len>targ_len: targ = F.pad(targ, (0,out_len-targ_len,0,0), value=pad_idx)
    return CrossEntropyFlat()(out, targ)
```

```python
xb,yb = next(iter(data.valid_dl))
```

```python
xb.shape
```

```python
yb.shape
```

```python
# output sequence length is the larger of the two inputs
rnn = Seq2SeqRNN(emb_enc, emb_dec, 256, max(xb.shape[1], yb.shape[1]))
```

```python
rnn
```

```python
# is only using up to 50% of GPU
# h = rnn.encoder(64, xb.cpu())
# try this to see if GPU usage increases
h = rnn.encoder(batch_size, xb.cpu())
```

```python
h.size()
```

### Custom accuracy metric for sequence to sequence models

```python
def seq2seq_acc(out, targ, pad_idx=1):
    bs,targ_len = targ.size()
    _,out_len,vs = out.size()
    if targ_len>out_len: out  = F.pad(out,  (0,0,0,targ_len-out_len,0,0), value=pad_idx)
    if out_len>targ_len: targ = F.pad(targ, (0,out_len-targ_len,0,0), value=pad_idx)
    out = out.argmax(2)
    return (out==targ).float().mean()
```

## Now train the Seq2seq RNN

```python
learn = None
rnn = None
def resetup():
    global learn
    global rnn
    del learn
    del rnn
    torch.cuda.empty_cache()
    gc.collect()
    rnn = Seq2SeqRNN(emb_enc, emb_dec, 256, max(xb.shape[1], yb.shape[1]))
    learn = Learner(data, rnn, loss_func=seq2seq_loss, metrics=seq2seq_acc)
```

```python
resetup()
```

```python
learn.lr_find()
```

```python
learn.recorder.plot()
```

### Work to determine optimal number of epochs

```python
# resetup()
# learn.fit_one_cycle(20, 5e-2) # swings from good (83) to bad (1) and back (unreliable) @ 128 batch size
```

```python
#resetup()
#learn.fit_one_cycle(5, 1e-2) # 51 - 63
#learn.fit_one_cycle(10, 1e-2) # 47 - 77
#learn.fit_one_cycle(20, 1e-2) # 42 - 74 (peak reached at 10, end is 65) @ 64 batch size
#learn.fit_one_cycle(20, 1e-2) # 23 - 77 @ 128 batch size
```

```python
# resetup()
# learn.fit_one_cycle(15, 1e-2) # after 46 - 78, peak reached around epoch 7, may evetually get better... perhaps more epochs
```

```python
# resetup()
# learn.fit_one_cycle(5, 5e-3) # 47 - 74
#learn.fit_one_cycle(10, 5e-3) # 26 - 81 @ 64 batch size
#learn.fit_one_cycle(20, 5e-3) # 45 - 85 @ 64 batch size
# learn.fit_one_cycle(25, 5e-3) # 32 - 87.6 @ 64 batch size
#learn.fit_one_cycle(10, 5e-3) # 38 - 74 @ 128 batch size
#learn.fit_one_cycle(20, 5e-3) # 19 - 84 @ 128 batch size
```

```python
resetup()
# learn.fit_one_cycle(5, 1e-3) # 40 - 72
# learn.fit_one_cycle(10, 1e-3) # 16 - 80 @ 64 batch size
# learn.fit_one_cycle(20, 1e-3) # 17 - 86 @ 64 batch size
learn.fit_one_cycle(25, 1e-3) # 17 - 87.9 @ 64 batch size
# learn.fit_one_cycle(10, 1e-3) # 16 - 68 @ 128 batch size
# learn.fit_one_cycle(20, 1e-3) # 16 - 83 @ 128 batch size
```

```python
# resetup()
# learn.fit_one_cycle(15, 5e-3) #
```

```python
# additional 5 after 5 results in similar, but not better performance
# learn.fit_one_cycle(5, 5e-3)
```

```python
# resetup()
# learn.fit_one_cycle(20, 5e-3) # 26 - 85
```

```python
# resetup()
# learn.fit_one_cycle(25, 5e-3) # 23 - 86
```

```python
learn.lr_find()
```

```python
learn.recorder.plot()
```

```python
learn.save('seq2seq_rnn')
```

```python
learn.load('seq2seq_rnn')
```

### Check how sequence genration is working so far

This seems to be a pretty poor prediction technique, though could be due to not enough model training.

Alternative techniques:

* Beam search - https://en.wikipedia.org/wiki/Beam_search
* top-k (right now sesarch is essential top-1; however most likely next word gets stuck and doesn't find best word for sequences of words.
* Nucleus  Sampling  - modified top-k approach

```python
def get_predictions(learn, ds_type=DatasetType.Valid):
    learn.model.eval()
    inputs, targets, outputs = [],[],[]
    with torch.no_grad():
        for xb,yb in progress_bar(learn.dl(ds_type)):
            out = learn.model(xb)
            for x,y,z in zip(xb,yb,out):
                inputs.append(learn.data.train_ds.x.reconstruct(x))
                targets.append(learn.data.train_ds.y.reconstruct(y))
                outputs.append(learn.data.train_ds.y.reconstruct(z.argmax(1)))
    return inputs, targets, outputs
```

```python
inputs, targets, outputs = get_predictions(learn)
```

```python
len(inputs)
```

```python
for n in range(10):
    print('input: ', inputs[n])
    print('target: ', targets[n])
    print('output: ', outputs[n])
    print('---------------------')
```

```python
inputs[700], targets[700], outputs[700]
```
