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

## Split text into lines/phrases for comparison of annotated vs unannotated.

Processes:
* Drop extra long (> 1000 chars) - this drops 123 lines of text or about 0.04% of the data
* Drop lines with only a single character - this drops and additional 151 lines of text
* For remaining lines, split into sentences using nltk.sent_tokenize() as it gives consistent results with both original text and annotated text. Now there are 47,643 lines of text.
* Remove duplicates and new sentences that were only a single character; end result is 37,880 comparable phrases


```python
from pathlib import Path
import pandas as pd
import os
import functools
import operator
import nltk
import seaborn as sns
sns.set()
```

```python
output_path = Path('../data')
ann_path = Path('../data/annotated_test')
text_path = Path('../data/orig_test')
ann_path = Path('../data/annotated')
text_path = Path('../data/orig')
paths = [ann_path, text_path]
```

Need to do the same for both annotated and original text.

```python
# need to load all text from ann_path and from text_path and split into sentences/lines
df = pd.DataFrame()

for p in paths:
    ann = {}
    file_counter = 0
    cols = [p.parts[-1] + '_source', p.parts[-1] + '_text']
    tdf = pd.DataFrame(columns = cols)
    for file in p.glob('*.txt'):
        file_counter += 1
        with open(file) as f:
            key = os.path.splitext(os.path.basename(file))[0]
            ann[key] = []
            lines = f.read().splitlines()
            # remove leading and trailing whitespace
            lines = map(str.strip, lines)
            # remove empty lines
            lines = list(filter(None, lines))
            ann[key].append(lines)
    for k in ann.keys():
        ann[k] = functools.reduce(operator.iconcat, ann[k], [])
        temp_df = pd.DataFrame(list(zip([k]*len(ann[k]), ann[k])),
                          columns = cols)
        tdf = tdf.append(temp_df)
    df = pd.concat([df, tdf], axis=1)
    
    print('Completed reading files from ', p)
    print('    Read', file_counter, 'files')
    print('    Read', tdf.shape[0], 'lines of text.')
```

```python
# this cell is for when using the smaller testing set
df.rename(columns={'annotated_test_text': 'annotated_text',
                   'annotated_test_source': 'annotated_source',
                   'orig_test_text': 'orig_text',
                   'orig_test_source': 'orig_source'
                  }, inplace=True)
```

```python
df.head()
```

```python
df.tail()
```

```python
df.shape
```

```python
sns.set(rc={'figure.figsize':(10,8)})
ax = sns.distplot(df['orig_text'].str.len(), hist=True, kde=False, rug=False)
ax.set_yscale('log')
```

Drop very long lines (these are paragraphs of text that have been run together)

```python
t_df = df[df.orig_text.str.len() <= 1000].copy()
```

```python
t_df.shape
```

```python
df.shape[0] - t_df.shape[0]
```

Drop single character lines - no meaning nor annotations are present

```python
f_df = t_df[t_df.annotated_text.str.len() > 1].copy()
```

```python
f_df.shape
```

```python
t_df.shape[0] - f_df.shape[0]
```

```python
f_df['orig_length'] = f_df['orig_text'].str.len()
f_df['annotated_length'] = f_df['annotated_text'].str.len()
f_df.sort_values('orig_length', inplace=True)
f_df.head()
```

```python
org_rows = []
ann_rows = []
max_diff = 0
min_diff = 0
mismatches = 0
new_rows = []
for row in f_df.itertuples():
    org_s = nltk.sent_tokenize(row.orig_text)
    ann_s = nltk.sent_tokenize(row.annotated_text)
    org_len = len(org_s)
    ann_len = len(ann_s)
    if(org_len != ann_len):
        mismatches += 1
        #print(row, org, ann)
        if (org_len - ann_len) > max_diff:
            max_diff = org_len - ann_len
        if (org_len - ann_len) < min_diff:
            min_diff = org_len - ann_len
    else:
        org_rows.append(org_s)
        ann_rows.append(ann_s)
        
        for n in range(org_len):
            new_rows.append({
                'orig_source': row.orig_source,
                'orig_text': org_s[n],
                'annotated_source': row.annotated_source,
                'annotated_text': ann_s[n]
            })
print('Total mismatches:', mismatches)
print('Biggest diff:', max_diff)
print('Biggest min:', min_diff)
```

```python
s_df = pd.DataFrame(new_rows)
s_df.shape
```

```python
# drop duplicates based on the annotated text column
s_df.drop_duplicates('annotated_text', inplace=True)
s_df = s_df[s_df.annotated_text.str.len() > 1].copy()
s_df.shape
```

```python
s_df['orig_length'] = s_df['orig_text'].str.len()
s_df['annotated_length'] = s_df['annotated_text'].str.len()
s_df.sort_values('orig_length', inplace=True)
s_df.head()
```

```python
s_df.to_csv(output_path/'processed_data.csv', index=False)
```
