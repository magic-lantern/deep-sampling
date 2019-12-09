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

# Annotation Converter

This notebook converts the original XML file format into plain text with markup inserted in it's proper place surrounded with tags delimeted with '{{' and '}}

```python
import xml.etree.ElementTree as ET
import os
from pathlib import Path
from html import unescape
```

```python
os.getcwd()
```

```python
# small sample for easier testing and manual validation
input_path = Path('../../training-PHI-Gold-Set1-test')
ann_path = Path('../data/annotated_test')
text_path = Path('../data/orig_test')
# full test data set
input_path = Path('../../training-PHI-Gold-Set1')
ann_path = Path('../data/annotated')
text_path = Path('../data/orig')

if not os.path.exists(ann_path):
    os.makedirs(ann_path)
    
if not os.path.exists(text_path):
    os.makedirs(text_path)
```

```python
# remove html encoded items and get rid of angled quotes
# it appears some files had html escaping run more than once, so have to unescape twice
def unencode(note):
    return unescape(unescape(note.replace('&#8220;', '"').replace('&#8221;', '"')))
```

```python
def insert_tags(text, tag, tag_type, start, end):
    TAG_START = '{{' # double curly braces do not appear in original text
    TAG_START1 = TAG_START
    TAG_START2 = TAG_START
    TAG_END   = '}} ' # need to include space here as without it tokenizer doesn't properly split
    LABEL_END = '/'
    TAG_SEPARATOR = ':'
    if tag == tag_type:
        combined = tag
    else:
        combined = tag + TAG_SEPARATOR + tag_type

    # need to check and add space before tag if doesnt exist already (if not beginning of line)
    if (not text[int(start) - 1].isspace()):
        TAG_START1 = ' ' + TAG_START1
#     # need to check and add space after tag if doesnt exist already (if not end of line)
#     if (not text[int(end)].isspace()):
#         print('end char:', text[int(end)])
#         print('end position:', int(end))
#         print('text:', text[int(end) - 10:int(end) + 10])
#         print('endtext------------------')
    
    offset = len(TAG_START1 + combined + TAG_END + TAG_START2 + LABEL_END + combined + TAG_END)
    return ((text[:int(start)] + 
            TAG_START1 + combined + TAG_END +
            text[int(start):int(end)] + 
            TAG_START2 + LABEL_END + combined + TAG_END +
            text[int(end):]), offset)
```

```python
def insert_space(text, start, end):
    TEXT_START = ''
    TEXT_END = ''
    # need to check and add space before tag if doesnt exist already (if not beginning of line)
    if (not text[int(start) - 1].isspace()):
        TEXT_START = ' '
#     # need to check and add space after tag if doesnt exist already (if not end of line)
#     if (not text[int(end)].isspace()):
#         print('end char:', text[int(end)])
#         print('end position:', int(end))
#         print('text:', text[int(end) - 10:int(end) + 10])
#         print('endtext------------------')
    
    offset = len(TEXT_START + TEXT_END)
    return ((text[:int(start)] + 
            TEXT_START +  text[int(start):int(end)] + TEXT_END +
            text[int(end):]), offset)
```

```python
for file in input_path.glob('*.xml'):
    with open(file) as f:
        tree = ET.parse(f)
        root = tree.getroot()
        note = root[0].text
        
        adj_note = note
        offset = 0
        for c in root.iter('TAGS'):
            for child in c:
                adj_note, new_offset = insert_space(adj_note,
                                               int(child.attrib['start']) + offset,
                                               int(child.attrib['end']) + offset)
                offset += new_offset
        adj_note = unencode(adj_note)
        
        # save original version of note
        new_file = os.path.splitext(os.path.basename(file))[0] + '.txt'
        with open(text_path / new_file, 'w') as o:
            o.write(adj_note)
        
        offset = 0
        for c in root.iter('TAGS'):
            for child in c:
                note, new_offset = insert_tags(note,
                                               child.tag,
                                               child.attrib['TYPE'],
                                               int(child.attrib['start']) + offset,
                                               int(child.attrib['end']) + offset)
                offset += new_offset
        note = unencode(note)
        # save annotated version of note text
        with open(ann_path / new_file, 'w') as o:
            o.write(note)
```

```python
os.path.splitext(os.path.basename(file))
```

## Rest of this notebook is just looking at 1 single file and checking processing steps

```python
with open('220-01.xml') as f:
    tree = ET.parse(f)
    root = tree.getroot()
    print(root)
```

```python
root.tag
```

```python
root.attrib
```

```python
for child in root:
    print(child.tag, child.attrib)
    print(type(child))
```

```python
note = root[0].text
```

```python
for c in root.iter('TAGS'):
    for child in c:
        print(child.tag, child.attrib)
```

```python
tag_start = '<'
tag_end = '>'
label_end = '/'
```

```python
new_note = root[0].text
offset = 0
for c in root.iter('TAGS'):
    for child in c:
        # print(child.attrib['start'], child.attrib['end'], child.tag, child.attrib['TYPE'])
        new_note, new_offset = insert_tags(new_note,
                                           child.tag,
                                           child.attrib['TYPE'],
                                           int(child.attrib['start']) + offset,
                                           int(child.attrib['end']) + offset)
        offset += new_offset
new_note
```

```python
int(child.attrib['start'])
```

```python
note
```

```python

```
