# Baseline Replication Results (SpanBERT on RECCON)

We include the replication results for SpanBERT on the RECCON dataset (Code: https://github.com/declare-lab/RECCON)

## Training (Causal Span Extraction - DailyDialog Dataset) 

### Fold 1

Namespace(batch_size=16, context=True, cuda=0, epochs=12, fold=1, lr=1e-05, model='span')

Postive Samples:
Exact Match: 648/1894 = 34.21%
Partial Match: 739/1894 = 39.02%
LCS F1 Score = 60.64%
SQuAD F1 Score = 60.88%
No Match: 507/1894 = 26.77%

Negative Samples
Inv F1 Score = 86.18%
----------------------------------------

### Fold 2

Namespace(batch_size=16, context=True, cuda=0, epochs=12, fold=2, lr=1e-05, model='span')

Postive Samples:
Exact Match: 780/1894 = 41.18%
Partial Match: 928/1894 = 49.0%
LCS F1 Score = 74.43%
SQuAD F1 Score = 74.8%
No Match: 186/1894 = 9.82%

Negative Samples
Inv F1 Score = 99.94%
----------------------------------------

### Fold 3

Namespace(batch_size=16, context=True, cuda=0, epochs=12, fold=3, lr=1e-05, model='span')

Postive Samples:
Exact Match: 802/1894 = 42.34%
Partial Match: 941/1894 = 49.68%
LCS F1 Score = 75.65%
SQuAD F1 Score = 76.08%
No Match: 151/1894 = 7.97%

Negative Samples
Inv F1 Score = 99.94%
----------------------------------------

## Evaluation (Causal Span Extraction - DailyDialog Dataset) 

### Fold 1

Namespace(batch_size=16, context=True, cuda=0, dataset='dailydialog', fold=1, model='span')

Postive Samples:
Exact Match: 648/1894 = 34.21%
Partial Match: 739/1894 = 39.02%
LCS F1 Score = 60.64%
SQuAD F1 Score = 60.88%
No Match: 507/1894 = 26.77%

Negative Samples:
Inv F1 Score = 86.18%

All Samples:
LCS F1 Score = 75.87%
SQuAD F1 Score = 75.93%
----------------------------------------

### Fold 2

Namespace(batch_size=16, context=True, cuda=0, dataset='dailydialog', fold=2, model='span')

Postive Samples:
Exact Match: 780/1894 = 41.18%
Partial Match: 928/1894 = 49.0%
LCS F1 Score = 74.43%
SQuAD F1 Score = 74.8%
No Match: 186/1894 = 9.82%

Negative Samples:
Inv F1 Score = 99.94%

All Samples:
LCS F1 Score = 92.3%
SQuAD F1 Score = 92.41%
----------------------------------------

### Fold 3

Namespace(batch_size=16, context=True, cuda=0, dataset='dailydialog', fold=3, model='span')

Postive Samples:
Exact Match: 802/1894 = 42.34%
Partial Match: 941/1894 = 49.68%
LCS F1 Score = 75.65%
SQuAD F1 Score = 76.08%
No Match: 151/1894 = 7.97%

Negative Samples:
Inv F1 Score = 99.94%

All Samples:
LCS F1 Score = 92.67%
SQuAD F1 Score = 92.8%
----------------------------------------

## Evaluation (Causal Span Extraction - IEMOCAP Dataset) 

### Fold 1

Namespace(batch_size=16, context=True, cuda=0, dataset='iemocap', fold=1, model='span')

Postive Samples:
Exact Match: 234/1080 = 21.67%
Partial Match: 227/1080 = 21.02%
LCS F1 Score = 37.41%
SQuAD F1 Score = 37.19%
No Match: 619/1080 = 57.31%

Negative Samples:
Inv F1 Score = 93.42%

All Samples:
LCS F1 Score = 87.36%
SQuAD F1 Score = 87.34%
----------------------------------------

### Fold 2

Namespace(batch_size=16, context=True, cuda=0, dataset='iemocap', fold=2, model='span')

Postive Samples:
Exact Match: 426/1080 = 39.44%
Partial Match: 507/1080 = 46.94%
LCS F1 Score = 74.99%
SQuAD F1 Score = 74.49%
No Match: 147/1080 = 13.61%

Negative Samples:
Inv F1 Score = 99.6%

All Samples:
LCS F1 Score = 96.74%
SQuAD F1 Score = 96.67%
----------------------------------------

### Fold 3

Namespace(batch_size=16, context=True, cuda=0, dataset='iemocap', fold=3, model='span')

Postive Samples:
Exact Match: 410/1080 = 37.96%
Partial Match: 518/1080 = 47.96%
LCS F1 Score = 74.19%
SQuAD F1 Score = 73.76%
No Match: 152/1080 = 14.07%

Negative Samples:
Inv F1 Score = 99.58%

All Samples:
LCS F1 Score = 96.72%
SQuAD F1 Score = 96.66%
----------------------------------------

Results may vary due to system dependencies. Experiments were carried out on GeForce RTX 3090 with PyTorch 1.7.0 and CuDA 11.0. Rest of the requirements and dependencies were kept the same as provided for RECCON at https://github.com/declare-lab/RECCON