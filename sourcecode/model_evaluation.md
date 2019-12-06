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
for n in range(10):
    print('input: ', inputs[n])
    print('target: ', targets[n])
    print('output: ', outputs[n])
    print('---------------------')
```

```python
inputs[700], targets[700], outputs[700]
```

```python

```
