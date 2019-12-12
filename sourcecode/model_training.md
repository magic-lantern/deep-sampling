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
%load_ext autoreload
%autoreload 1
```

```python
from fastai.text import *
from seq2seq import DataBunch, Seq2SeqTextList, TeacherForcing
import pandas as pd

# in develoment library, see https://github.com/facebookresearch/fastText to download, how to build, etc.
import fasttext as ft

import custom_functions as c
%aimport custom_functions
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
# attention model requires smaller batch size to fit in GPU memory (16GB)
batch_size = 32
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
src = (Seq2SeqTextList.from_df(df, path = path, cols='orig_text')
       .split_by_rand_pct(seed=seed)
       .label_from_df(cols='annotated_text', label_cls=TextList))
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

### Custom loss metric for sequence to sequence models

The `seq2seq_loss` function has been moved to custom_functions.py for reusability


### Build RNN

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
rnn = c.Seq2SeqRNN(emb_enc, emb_dec, 256, max(xb.shape[1], yb.shape[1]))
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

The `seq2seq_acc` function has been moved to custom_functions.py for reusability


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
    rnn = c.Seq2SeqRNN(emb_enc, emb_dec, 256, max(xb.shape[1], yb.shape[1]))
    learn = Learner(data, rnn, loss_func=c.seq2seq_loss, metrics=c.seq2seq_acc)
```

```python
xb.shape[1]
```

```python
yb.shape[1]
```

```python
max(xb.shape[1], yb.shape[1])
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

**Note**

Found 1e-3 has best results with current data, batch size of 64, and running on GPU

```python
resetup()
learn.fit_one_cycle(5, 1e-3) # 40 - 72
```

```python
inputs, targets, outputs = c.get_predictions(learn)
for n in range(5):
    print('input: ', inputs[n])
    print('target: ', targets[n])
    print('output: ', outputs[n])
    print('---------------------')
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

```python
class TeacherForcing(LearnerCallback):
    def __init__(self, learn, end_epoch):
        super().__init__(learn)
        self.end_epoch = end_epoch
    
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        if train: return {'last_input': [last_input, last_target]}
    
    def on_epoch_begin(self, epoch, **kwargs):
        self.learn.model.pr_force = max(0, 1 - epoch/self.end_epoch)
        print('force probability', self.learn.model.pr_force)
```

```python
class Seq2SeqRNN_tf(nn.Module):
    def __init__(self, emb_enc, emb_dec, nh, out_sl, nl=2, bos_idx=0, pad_idx=1):
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
        self.pr_force = 0.
        
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
            
    def forward(self, inp, targ=None):
        bs, sl = inp.size()
        h = self.encoder(bs, inp)
        dec_inp = inp.new_zeros(bs).long() + self.bos_idx
        
        res = []
        for i in range(self.out_sl):
            h, outp = self.decoder(dec_inp, h)
            res.append(outp)
            dec_inp = outp.max(1)[1]
            if (dec_inp==self.pad_idx).all(): break
            if (targ is not None) and (random.random()<self.pr_force):
                if i>=targ.shape[1]: continue
                dec_inp = targ[:,i]
        return torch.stack(res, dim=1)

    def initHidden(self, bs): return one_param(self).new_zeros(self.nl, bs, self.nh)
```

```python
class NGram():
    def __init__(self, ngram, max_n=5000): self.ngram,self.max_n = ngram,max_n
    def __eq__(self, other):
        if len(self.ngram) != len(other.ngram): return False
        return np.all(np.array(self.ngram) == np.array(other.ngram))
    def __hash__(self): return int(sum([o * self.max_n**i for i,o in enumerate(self.ngram)]))

def get_grams(x, n, max_n=5000):
    return x if n==1 else [NGram(x[i:i+n], max_n=max_n) for i in range(len(x)-n+1)]

def get_correct_ngrams(pred, targ, n, max_n=5000):
    pred_grams,targ_grams = get_grams(pred, n, max_n=max_n),get_grams(targ, n, max_n=max_n)
    pred_cnt,targ_cnt = Counter(pred_grams),Counter(targ_grams)
    return sum([min(c, targ_cnt[g]) for g,c in pred_cnt.items()]),len(pred_grams)
```

```python
class CorpusBLEU(Callback):
    def __init__(self, vocab_sz):
        self.vocab_sz = vocab_sz
        self.name = 'bleu'
    
    def on_epoch_begin(self, **kwargs):
        self.pred_len,self.targ_len,self.corrects,self.counts = 0,0,[0]*4,[0]*4
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        last_output = last_output.argmax(dim=-1)
        for pred,targ in zip(last_output.cpu().numpy(),last_target.cpu().numpy()):
            self.pred_len += len(pred)
            self.targ_len += len(targ)
            for i in range(4):
                c,t = get_correct_ngrams(pred, targ, i+1, max_n=self.vocab_sz)
                self.corrects[i] += c
                self.counts[i]   += t
    
    def on_epoch_end(self, last_metrics, **kwargs):
        precs = [c/t for c,t in zip(self.corrects,self.counts)]
        len_penalty = exp(1 - self.targ_len/self.pred_len) if self.pred_len < self.targ_len else 1
        bleu = len_penalty * ((precs[0]*precs[1]*precs[2]*precs[3]) ** 0.25)
        return add_metrics(last_metrics, bleu)
```

```python
learn_tf = None
rnn_tf = None
def setup_tf(end_epochs=5):
    global learn_tf
    global rnn_tf
    del learn_tf
    del rnn_tf
    torch.cuda.empty_cache()
    gc.collect()
    rnn_tf = Seq2SeqRNN_tf(emb_enc, emb_dec, 256, max(xb.shape[1], yb.shape[1]))
    learn_tf = Learner(data, rnn_tf,
                       loss_func=c.seq2seq_loss,
                       metrics=[c.seq2seq_acc, CorpusBLEU(len(data.y.vocab.itos))],
                       callback_fns=partial(TeacherForcing, end_epoch=end_epochs))
```

```python
rnn_tf = Seq2SeqRNN_tf(emb_enc, emb_dec, 256, max(xb.shape[1], yb.shape[1]))

learn_tf = Learner(data, rnn_tf,
                loss_func=c.seq2seq_loss,
                metrics=[c.seq2seq_acc, CorpusBLEU(len(data.y.vocab.itos))],
                callback_fns=partial(TeacherForcing, end_epoch=3))
```

```python
learn_tf.lr_find()
```

```python
learn_tf.recorder.plot()
```

```python
learn_tf.fit_one_cycle(5, 3e-3)
```

```python
setup_tf(5)
learn_tf.lr_find()
learn_tf.recorder.plot()
```

```python
learn_tf.fit_one_cycle(5, 3e-3)
```

```python
setup_tf(5)
learn_tf.fit_one_cycle(5, 3e-3)
```

```python
# not sure why, but with teacher forcing, can't get last few prediction values
def get_predictions(learn, ds_type=DatasetType.Valid, trim=False):
    learn.model.eval()
    inputs, targets, outputs = [],[],[]
    with torch.no_grad():
        count = 1
        for xb,yb in progress_bar(learn.dl(ds_type)):
            out = learn.model(xb)
            for x,y,z in zip(xb,yb,out):
                inputs.append(learn.data.train_ds.x.reconstruct(x))
                targets.append(learn.data.train_ds.y.reconstruct(y))
                outputs.append(learn.data.train_ds.y.reconstruct(z.argmax(1)))
            print(count)
            count += 1
    return inputs, targets, outputs
```

```python
inputs, targets, outputs = c.get_predictions(learn_tf)
for n in range(5):
    print('input: ', inputs[n])
    print('target: ', targets[n])
    print('output: ', outputs[n])
    print('---------------------')
```

```python
class Seq2SeqRNN_attn(nn.Module):
    def __init__(self, emb_enc, emb_dec, nh, out_sl, nl=2, bos_idx=0, pad_idx=1):
        super().__init__()
        self.nl,self.nh,self.out_sl,self.pr_force = nl,nh,out_sl,1
        self.bos_idx,self.pad_idx = bos_idx,pad_idx
        self.emb_enc,self.emb_dec = emb_enc,emb_dec
        self.emb_sz_enc,self.emb_sz_dec = emb_enc.embedding_dim,emb_dec.embedding_dim
        self.voc_sz_dec = emb_dec.num_embeddings
                 
        self.emb_enc_drop = nn.Dropout(0.15)
        self.gru_enc = nn.GRU(self.emb_sz_enc, nh, num_layers=nl, dropout=0.25, 
                              batch_first=True, bidirectional=True)
        self.out_enc = nn.Linear(2*nh, self.emb_sz_dec, bias=False)
        
        self.gru_dec = nn.GRU(self.emb_sz_dec + 2*nh, self.emb_sz_dec, num_layers=nl,
                              dropout=0.1, batch_first=True)
        self.out_drop = nn.Dropout(0.35)
        self.out = nn.Linear(self.emb_sz_dec, self.voc_sz_dec)
        self.out.weight.data = self.emb_dec.weight.data
        
        self.enc_att = nn.Linear(2*nh, self.emb_sz_dec, bias=False)
        self.hid_att = nn.Linear(self.emb_sz_dec, self.emb_sz_dec)
        self.V =  self.init_param(self.emb_sz_dec)
        
    def encoder(self, bs, inp):
        h = self.initHidden(bs)
        emb = self.emb_enc_drop(self.emb_enc(inp))
        enc_out, hid = self.gru_enc(emb, 2*h)
        
        pre_hid = hid.view(2, self.nl, bs, self.nh).permute(1,2,0,3).contiguous()
        pre_hid = pre_hid.view(self.nl, bs, 2*self.nh)
        hid = self.out_enc(pre_hid)
        return hid,enc_out
    
    def decoder(self, dec_inp, hid, enc_att, enc_out):
        hid_att = self.hid_att(hid[-1])
        # we have put enc_out and hid through linear layers
        u = torch.tanh(enc_att + hid_att[:,None])
        # we want to learn the importance of each time step
        attn_wgts = F.softmax(u @ self.V, 1)
        # weighted average of enc_out (which is the output at every time step)
        ctx = (attn_wgts[...,None] * enc_out).sum(1)
        emb = self.emb_dec(dec_inp)
        # concatenate decoder embedding with context (we could have just
        # used the hidden state that came out of the decoder, if we weren't
        # using attention)
        outp, hid = self.gru_dec(torch.cat([emb, ctx], 1)[:,None], hid)
        outp = self.out(self.out_drop(outp[:,0]))
        return hid, outp
        
    def show(self, nm,v):
        if False: print(f"{nm}={v[nm].shape}")
        
    def forward(self, inp, targ=None):
        bs, sl = inp.size()
        hid,enc_out = self.encoder(bs, inp)
#        self.show("hid",vars())
        dec_inp = inp.new_zeros(bs).long() + self.bos_idx
        enc_att = self.enc_att(enc_out)
        
        res = []
        for i in range(self.out_sl):
            hid, outp = self.decoder(dec_inp, hid, enc_att, enc_out)
            res.append(outp)
            dec_inp = outp.max(1)[1]
            if (dec_inp==self.pad_idx).all(): break
            if (targ is not None) and (random.random()<self.pr_force):
                if i>=targ.shape[1]: continue
                dec_inp = targ[:,i]
        return torch.stack(res, dim=1)

    def initHidden(self, bs): return one_param(self).new_zeros(2*self.nl, bs, self.nh)
    def init_param(self, *sz): return nn.Parameter(torch.randn(sz)/math.sqrt(sz[0]))
```

```python
torch.cuda.empty_cache()
gc.collect()

```

```python
learn_attn = None
rnn_attn = None
def setup_attn(end_epochs=5):
    global learn_attn
    global rnn_attn
    del learn_attn
    del rnn_attn
    torch.cuda.empty_cache()
    gc.collect()
    rnn_attn = Seq2SeqRNN_attn(emb_enc, emb_dec, 256, max(xb.shape[1], yb.shape[1]))
    learn_attn = Learner(data, rnn_attn,
                       loss_func=c.seq2seq_loss,
                       metrics=[c.seq2seq_acc, CorpusBLEU(len(data.y.vocab.itos))],
                       callback_fns=partial(TeacherForcing, end_epoch=end_epochs))
```

```python
rnn_attn
```

```python
learn_attn
```

```python
setup_attn()
learn_attn.lr_find()
learn_attn.recorder.plot()
```

```python
setup_attn()
learn_attn.fit_one_cycle(15, 3e-3)
```

```python
inputs, targets, outputs = c.get_predictions(learn_attn)
for n in range(5):
    print('input: ', inputs[n])
    print('target: ', targets[n])
    print('output: ', outputs[n])
    print('---------------------')
```

```python

```
