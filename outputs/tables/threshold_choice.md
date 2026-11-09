| model               |   threshold | rule                                   |   val_fpr |   val_recall |   val_f1 |   test_fpr |   test_recall |
|:--------------------|------------:|:---------------------------------------|----------:|-------------:|---------:|-----------:|--------------:|
| Logistic regression |      0.8027 | max F1 subject to validation FPR <= 5% |         0 |       0.9979 |    0.999 |  5.424e-05 |        0.999  |
| Random Forest       |      0.9727 | max F1 subject to validation FPR <= 5% |         0 |       0.998  |    0.999 |  0         |        0.9986 |
| Gradient Boosting   |      1      | max F1 subject to validation FPR <= 5% |         0 |       0.9979 |    0.999 |  0         |        0.9984 |