---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

Split text into groups for comparison of annotated vs unannotated.

To Do:
* Drop extra long (> 1000 chars)
* For rest, should long lines be split into sentences?
* Need to keep track of source file

```python
from fastai.text import *
import spacy
from spacy.symbols import ORTH, NORM
import seaborn as sns
sns.set()
```

```python
ann_path = Path('../data/annotated')
text_path = Path('../data/orig_text')
```

```python
# need to load all text from ann_path and from text_path and split into sentences/lines
ann = []
for file in ann_path.glob('*.txt'):
    with open(file) as f:
        lines = f.read().splitlines()
        # remove leading and trailing whitespace
        lines = map(str.strip, lines)
        # remove empty lines
        lines = list(filter(None, lines))
        ann.append(lines)
```

```python
ann = functools.reduce(operator.iconcat, ann, [])
```

```python
len(ann)
```

```python
# need to load all text from ann_path and from text_path and split into sentences/lines
texts = []
for file in text_path.glob('*.txt'):
    with open(file) as f:
        lines = f.read().splitlines()
        # remove leading and trailing whitespace
        lines = map(str.strip, lines)
        # remove empty lines
        lines = list(filter(None, lines))
        texts.append(lines)
```

```python
texts = functools.reduce(operator.iconcat, texts, [])
```

```python
len(texts)
```

```python
texts_lens = [len(i) for i in texts]
sns.set(rc={'figure.figsize':(10,10)})
ax = sns.distplot(texts_lens, hist=True, kde=False, rug=False)
ax.set_yscale('log')
```

```python
sum(len(t) <= 1000 for t in texts) / len(texts)
```

```python
df = pd.DataFrame(list(zip(texts, ann)), 
               columns =['original', 'annotated']) 
df.head()
```

```python
df[df.original.str.len() <= 1000]

#df['a'].astype(str).str.len()
```

```python
for i in range(0,len(ann)):
    print(ann[i])
    print(text[i])
    print('----------')
    if i > 10:
        break
```

```python
text.sort(key = len)
text[-1]
```

```python
max_len = 0
longest = ''
for p in ann:
    if len(p) > max_len:
        max_len = len(p)
        longest = p
ann.sort(key = len)
ann[-1]
```
