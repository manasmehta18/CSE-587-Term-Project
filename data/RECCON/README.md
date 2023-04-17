## Additional Instructions
This folder contains the original version of [RECCON](https://github.com/declare-lab/RECCON/tree/main/data/original_annotation) dataset. However, we found some annotation errors in it and fixed them accordingly.

### 1. Causal Span Boundary

There are some errors with the causal span boundary, for example:
```bash
## causal utterance:
I have bought a pencile last Friday.

## original annotation:
I have bought a pen

## fixed annotation:
I have bought a pencil
```

### 2. Emotion Labels
Inconsistent emotion labels, for example:
```bash
#  original                 ->  fixed
happy, happiness, happines  ->  happy 
anger, angery               ->  anger
suprise, surprised          ->  surprise
```