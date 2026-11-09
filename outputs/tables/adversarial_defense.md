| attack                                     |   recall_baseline_model |   recall_after_adversarial_training |   recall_recovered |
|:-------------------------------------------|------------------------:|------------------------------------:|-------------------:|
| A0 No attack (clean scan flows)            |               0.9986    |                             0.9979  |         -0.0006616 |
| A1 Timing evasion, 10x slower              |               0.771     |                             0.9946  |          0.2236    |
| A1 Timing evasion, 100x slower             |               0.02551   |                             0.725   |          0.6995    |
| A1 Timing evasion, 1000x slower            |               7.351e-05 |                             0.02389 |          0.02382   |
| A2 Payload padding, 24 bytes               |               0         |                             0.987   |          0.987     |
| A2 Payload padding, 64 bytes               |               0         |                             0       |          0         |
| A3 TCP window reshaping                    |               0.7489    |                             0.9909  |          0.242     |
| A4 Handshake mimicry (connect scan)        |               0         |                             0.998   |          0.998     |
| A5 Combined (A1 100x + A2 + A3 + A4)       |               0         |                             0.9982  |          0.9982    |
| Clean benign traffic (false-positive rate) |               0         |                             0       |          0         |