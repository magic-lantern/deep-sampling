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

```python
import re
import spacy
from spacy.tokenizer import Tokenizer
import nltk
```

```python
text1_o = 'Dear Dr. Craft:'
text1_a = 'Dear Dr. {{NAME:DOCTOR}}Craft{{/NAME:DOCTOR}}:'
text2_o = 'In summary, this is a very pleasant 61-year-old gentleman who has a history of a large anteroseptal MI approximately six years ago, had stenting his LAD in 2076, and now has a recurrence of what sounds like possible angina and a positive stress test has a decrease in LV function, increase in LV aneurysmal dilatation, and ischemia in the anterior distribution. There is a discrepancy between the ejection fraction on the echo and I suspect that has to do with differences in technique. I have talked with the patient about cardiac catheterization and I would like to get him in for that as soon as possible so we can assess the degree of coronary disease. Hopefully it is amenable to percutaneous angioplasty, but I am somewhat concerned about the left ventricular aneurysm. We will arrange for cardiac catheterization either next week or the following week and I will plan to see him back here in the office approximately a month which will be after his catheterization and hopeful angioplasty.'
text2_a = 'In summary, this is a very pleasant {{AGE}}61{{/AGE}}-year-old gentleman who has a history of a large anteroseptal MI approximately six years ago, had stenting his LAD in {{DATE}}2076{{/DATE}}, and now has a recurrence of what sounds like possible angina and a positive stress test has a decrease in LV function, increase in LV aneurysmal dilatation, and ischemia in the anterior distribution. There is a discrepancy between the ejection fraction on the echo and I suspect that has to do with differences in technique. I have talked with the patient about cardiac catheterization and I would like to get him in for that as soon as possible so we can assess the degree of coronary disease. Hopefully it is amenable to percutaneous angioplasty, but I am somewhat concerned about the left ventricular aneurysm. We will arrange for cardiac catheterization either next week or the following week and I will plan to see him back here in the office approximately a month which will be after his catheterization and hopeful angioplasty.'
comps1 = [text1_o, text1_a]
comps2 = [text2_o, text2_a]
```

```python
for t in comps1:
    sent_text = nltk.sent_tokenize(t)
    lines = 1
    for sentence in sent_text:
        print(lines, sentence)
        lines += 1
    print('----------------------')
```

```python
nlp = spacy.load("en_core_web_sm")

for t in comps1:
    sent_text = nlp(t)
    lines = 1
    for sentence in sent_text.sents:
        print(lines, sentence)
        lines += 1
    print('----------------------')
```

<!-- #region -->
Comparison between sentence segmentation with NLTK and SpaCy. For this project, key is to get consistent segmentation between annotation and non-annotation. NLTK seems to fit the use case best

```python
max_diff = 0
min_diff = 0
mismatches = 0
for row in f_df.itertuples():
    org = len(nltk.sent_tokenize(row.original))
    ann = len(nltk.sent_tokenize(row.annotated))
    if(org != ann):
        mismatches += 1
        print(row, org, ann)
        if (org - ann) > max_diff:
            max_diff = org - ann
        if (org - ann) < min_diff:
            min_diff = org - ann
print('Total mismatches:', mismatches)
print('Biggest diff:', max_diff)
print('Biggest min:', min_diff)
```

    Total mismatches: 28
    Biggest diff: 1
    Biggest min: -2

```python
max_diff = 0
min_diff = 0
mismatches = 0
for row in f_df.itertuples():
    org = len(list(nlp(row.original).sents))
    ann = len(list(nlp(row.annotated).sents))
    if(org != ann):
        mismatches += 1
        #print(row, org, ann)
        if (org - ann) > max_diff:
            max_diff = org - ann
        if (org - ann) < min_diff:
            min_diff = org - ann
print('Total mismatches:', mismatches)
print('Biggest diff:', max_diff)
print('Biggest min:', min_diff)
```

    Total mismatches: 2689
    Biggest diff: 69
    Biggest min: -8

<!-- #endregion -->

```python
#infix_re = re.compile(r'''({{/\w+)|({{)''')
infix_re = re.compile(r'''({{/\w+:\w+}})|({{)''')
suffix_re = re.compile(r'''(}}[)]*\.)$''')
suffix_re = re.compile(r'''(}}\.)$''')

def custom_tokenizer(nlp):
  return Tokenizer(nlp.vocab, 
                   suffix_search=suffix_re.search)

# def custom_tokenizer(nlp):
#   return Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)

text = text3

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = custom_tokenizer(nlp)
doc = nlp(text)
print('Total # of Sentences:', len(list(doc.sents)))
counter = 1
for sent in doc.sents:
    print(counter, ':', sent.text)
    print('------------')
    counter+=1
```

```python

```
