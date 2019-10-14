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

**To Do:**


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
input_path = Path('../../training-PHI-Gold-Set1')
output_path = Path('../data/converted')
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
    TAG_END   = '}}'
    LABEL_END = '/'
    TAG_SEPARATOR = ':'
    if tag == tag_type:
        combined = tag
    else:
        combined = tag + TAG_SEPARATOR + tag_type
    offset = len(TAG_START + combined + TAG_END + TAG_START + LABEL_END + combined + TAG_END)
    return ((text[:int(start)] + 
            TAG_START + combined + TAG_END +
            text[int(start):int(end)] + 
            TAG_START + LABEL_END + combined + TAG_END +
            text[int(end):]), offset)
```

```python
if not os.path.exists(output_path):
    os.makedirs(output_path)
new_file = ''
for file in input_path.glob('*.xml'):
    with open(file) as f:
        tree = ET.parse(f)
        root = tree.getroot()
        note = root[0].text
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
        
        new_file = os.path.splitext(os.path.basename(file))[0] + '.txt'
        print('output_path / new_file: ', output_path / new_file)
        with open(output_path / new_file, 'w') as o:
            o.write(note)
        break
```

```python
os.path.splitext(os.path.basename(file))
```

## Rest of notebook is just looking at 1 single file and checking processing steps

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
