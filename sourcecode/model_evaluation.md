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

# Evaluate Model for annotation


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

# custom functions from model_training.ipynb
# use autoreload extension to auto load when changes occur
import custom_functions as c
%aimport custom_functions
```

```python
# to help with getting consistent results
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
# not sure if these are needed or not
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
```

```python
path = Path('../data')
in_data_file = path/'processed_data.csv'
databunch_file = path/'databunch.pkl'
```

```python
# this takes about 1 minute to run
learn = c.get_learner()
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
for n in range(5):
    print('input: ', inputs[n])
    print('target: ', targets[n])
    print('output: ', outputs[n])
    print('---------------------')
```

```python
inputs[700], targets[700], outputs[700]
```

```python
def preds_acts(learn, ds_type=DatasetType.Valid):
    "Same as `get_predictions` but also returns non-reconstructed activations"
    learn.model.eval()
    ds = learn.data.train_ds
    rxs,rys,rzs,xs,ys,zs = [],[],[],[],[],[] # 'r' == 'reconstructed'
    with torch.no_grad():
        for xb,yb in progress_bar(learn.dl(ds_type)):
            out = learn.model(xb)
            for x,y,z in zip(xb,yb,out):
                rxs.append(ds.x.reconstruct(x))
                rys.append(ds.y.reconstruct(y))
                preds = z.argmax(1)
                rzs.append(ds.y.reconstruct(preds))
                for a,b in zip([xs,ys,zs],[x,y,z]): a.append(b)
    return rxs,rys,rzs,xs,ys,zs
```

```python
rxs,rys,rzs,xs,ys,zs = preds_acts(learn)
```

```python
def select_topk(outp, k=5):
    probs = F.softmax(outp,dim=-1)
    vals,idxs = probs.topk(k, dim=-1)
    return idxs[torch.randint(k, (1,))]
```

```python
def decode(self, inp):
    inp = inp[None]
    bs, sl = inp.size()
    hid,enc_out = self.encoder(bs, inp)
    dec_inp = inp.new_zeros(bs).long() + self.bos_idx
    enc_att = self.enc_att(enc_out)

    res = []
    for i in range(self.out_sl):
        hid, outp = self.decoder(dec_inp, hid, enc_att, enc_out)
        #dec_inp = select_nucleus(outp[0], p=0.3)
        dec_inp = select_topk(outp[0], k=2)
        res.append(dec_inp)
        if (dec_inp==self.pad_idx).all(): break
    return torch.cat(res)
```

```python
def predict_with_decode(learn, x, y):
    learn.model.eval()
    ds = learn.data.train_ds
    with torch.no_grad():
        out = decode(learn.model, x)
        rx = ds.x.reconstruct(x)
        ry = ds.y.reconstruct(y)
        rz = ds.y.reconstruct(out)
    return rx,ry,rz
```

```python
rxs,rys,rzs,xs,ys,zs = preds_acts(learn)

for n in range(5):
    rx,ry,rz = rxs[n],rys[n],rzs[n]
    x,y,z = xs[n],ys[n],zs[n]
    rx,ry,rz = predict_with_decode(learn, x, y)
    print('input: ', rx)
    print('target: ', ry)
    print('output: ', rz)
    print('---------------------')
```

```python

```
