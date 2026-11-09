| attack                                     |   recall_baseline_model |   recall_after_adversarial_training |   recall_recovered |
|:-------------------------------------------|------------------------:|------------------------------------:|-------------------:|
| A0 No attack (clean scan flows)            |                0.999    |                             0.9997  |          0.0007351 |
| A1 Timing evasion, 10x slower              |                0.98     |                             0.9994  |          0.01941   |
| A1 Timing evasion, 100x slower             |                0.06851  |                             0.9766  |          0.9081    |
| A1 Timing evasion, 1000x slower            |                0.003969 |                             0.05256 |          0.04859   |
| A2 Payload padding, 24 bytes               |                0.000441 |                             0.9999  |          0.9995    |
| A2 Payload padding, 64 bytes               |                0        |                             0.9946  |          0.9946    |
| A3 TCP window reshaping                    |                0.9963   |                             0.9993  |          0.00294   |
| A4 Handshake mimicry (connect scan)        |                0.009335 |                             0.9999  |          0.9906    |
| A5 Combined (A1 100x + A2 + A3 + A4)       |                0        |                             0.9997  |          0.9997    |
| Clean benign traffic (false-positive rate) |                0        |                             0       |          0         |