| feature                     | decision                           |   train_auc | reason                                                                          |
|:----------------------------|:-----------------------------------|------------:|:--------------------------------------------------------------------------------|
| Bwd PSH Flags               | reduced set: excluded (constant)   |    nan      | Single value in raw file; kept in all-features sets                             |
| Fwd URG Flags               | reduced set: excluded (constant)   |    nan      | Single value in raw file; kept in all-features sets                             |
| Bwd URG Flags               | reduced set: excluded (constant)   |    nan      | Single value in raw file; kept in all-features sets                             |
| CWE Flag Count              | reduced set: excluded (constant)   |    nan      | Single value in raw file; kept in all-features sets                             |
| Fwd Avg Bytes/Bulk          | reduced set: excluded (constant)   |    nan      | Single value in raw file; kept in all-features sets                             |
| Fwd Avg Packets/Bulk        | reduced set: excluded (constant)   |    nan      | Single value in raw file; kept in all-features sets                             |
| Fwd Avg Bulk Rate           | reduced set: excluded (constant)   |    nan      | Single value in raw file; kept in all-features sets                             |
| Bwd Avg Bytes/Bulk          | reduced set: excluded (constant)   |    nan      | Single value in raw file; kept in all-features sets                             |
| Bwd Avg Packets/Bulk        | reduced set: excluded (constant)   |    nan      | Single value in raw file; kept in all-features sets                             |
| Bwd Avg Bulk Rate           | reduced set: excluded (constant)   |    nan      | Single value in raw file; kept in all-features sets                             |
| Fwd Header Length.1         | reduced set: excluded (exact copy) |    nan      | Identical to another column; kept in all-features sets                          |
| Subflow Fwd Packets         | reduced set: excluded (exact copy) |    nan      | Identical to another column; kept in all-features sets                          |
| Subflow Bwd Packets         | reduced set: excluded (exact copy) |    nan      | Identical to another column; kept in all-features sets                          |
| Subflow Fwd Bytes           | reduced set: excluded (exact copy) |    nan      | Identical to another column; kept in all-features sets                          |
| Subflow Bwd Bytes           | reduced set: excluded (exact copy) |    nan      | Identical to another column; kept in all-features sets                          |
| Avg Fwd Segment Size        | reduced set: excluded (exact copy) |    nan      | Identical to another column; kept in all-features sets                          |
| Fwd PSH Flags               | reduced set: excluded (exact copy) |    nan      | Identical to another column; kept in all-features sets                          |
| ECE Flag Count              | reduced set: excluded (exact copy) |    nan      | Identical to another column; kept in all-features sets                          |
| Bwd Packets/s               | kept                               |      0.9375 | not redundant                                                                   |
| Fwd Packet Length Mean      | kept                               |      0.931  | not redundant                                                                   |
| Total Length of Fwd Packets | kept                               |      0.9309 | not redundant                                                                   |
| Fwd Packet Length Max       | kept                               |      0.9307 | not redundant                                                                   |
| Flow Duration               | kept                               |      0.9007 | not redundant                                                                   |
| Packet Length Mean          | kept                               |      0.9007 | not redundant                                                                   |
| Flow IAT Max                | kept                               |      0.8975 | not redundant                                                                   |
| Average Packet Size         | dropped                            |      0.8944 | |r| = 0.993 with kept feature 'Packet Length Mean' (AUC 0.901)                  |
| Flow Packets/s              | kept                               |      0.8869 | not redundant                                                                   |
| Fwd Packets/s               | dropped                            |      0.8796 | |r| = 0.951 with kept feature 'Flow Packets/s' (AUC 0.887)                      |
| PSH Flag Count              | kept                               |      0.8767 | not redundant                                                                   |
| Total Fwd Packets           | kept                               |      0.8708 | not redundant                                                                   |
| Fwd IAT Total               | dropped                            |      0.8673 | |r| = 0.999 with kept feature 'Flow Duration' (AUC 0.901)                       |
| Fwd IAT Max                 | dropped                            |      0.8671 | |r| = 0.997 with kept feature 'Flow IAT Max' (AUC 0.898)                        |
| Fwd IAT Mean                | kept                               |      0.867  | not redundant                                                                   |
| Fwd IAT Min                 | dropped                            |      0.8635 | |r| = 0.966 with kept feature 'Fwd IAT Mean' (AUC 0.867)                        |
| Flow IAT Mean               | dropped                            |      0.8506 | |r| = 0.982 with kept feature 'Fwd IAT Mean' (AUC 0.867)                        |
| act_data_pkt_fwd            | kept                               |      0.8477 | not redundant                                                                   |
| Max Packet Length           | kept                               |      0.8402 | not redundant                                                                   |
| Flow IAT Std                | dropped                            |      0.8276 | |r| = 0.953 with kept feature 'Fwd IAT Mean' (AUC 0.867)                        |
| Init_Win_bytes_forward      | kept                               |      0.808  | not redundant                                                                   |
| Bwd Header Length           | kept                               |      0.8007 | not redundant                                                                   |
| init_win_bwd_missing        | kept                               |      0.7964 | not redundant                                                                   |
| Bwd IAT Total               | dropped                            |      0.7914 | |r| = 0.988 with kept feature 'Flow Duration' (AUC 0.901)                       |
| Bwd IAT Max                 | dropped                            |      0.7912 | |r| = 0.958 with kept feature 'Flow IAT Max' (AUC 0.898)                        |
| Bwd IAT Mean                | dropped                            |      0.7911 | |r| = 0.966 with kept feature 'Fwd IAT Mean' (AUC 0.867)                        |
| Bwd IAT Min                 | kept                               |      0.7894 | not redundant                                                                   |
| min_seg_size_forward        | kept                               |      0.7731 | not redundant                                                                   |
| Packet Length Std           | dropped                            |      0.7675 | |r| = 0.972 with kept feature 'Max Packet Length' (AUC 0.840)                   |
| Packet Length Variance      | kept                               |      0.7675 | not redundant                                                                   |
| Fwd Header Length           | dropped                            |      0.761  | |r| = 0.962 with kept feature 'Total Fwd Packets' (AUC 0.871)                   |
| Total Length of Bwd Packets | dropped                            |      0.7609 | |r| = 0.957 with kept feature 'Bwd Header Length' (AUC 0.801)                   |
| Bwd Packet Length Max       | kept                               |      0.7534 | not redundant                                                                   |
| Bwd Packet Length Mean      | kept                               |      0.7507 | not redundant                                                                   |
| Avg Bwd Segment Size        | dropped                            |      0.7507 | |r| = 1.000 with kept feature 'Bwd Packet Length Mean' (AUC 0.751)              |
| init_win_fwd_missing        | kept                               |      0.7382 | not redundant                                                                   |
| Fwd Packet Length Min       | kept                               |      0.7347 | not redundant                                                                   |
| Total Backward Packets      | dropped                            |      0.7329 | |r| = 0.976 with kept feature 'Bwd Header Length' (AUC 0.801)                   |
| Min Packet Length           | kept                               |      0.7289 | not redundant                                                                   |
| Flow Bytes/s                | kept                               |      0.7186 | not redundant                                                                   |
| Fwd IAT Std                 | kept                               |      0.6728 | not redundant                                                                   |
| Fwd Packet Length Std       | kept                               |      0.6601 | not redundant                                                                   |
| Down/Up Ratio               | kept                               |      0.656  | not redundant                                                                   |
| Flow IAT Min                | kept                               |      0.6479 | not redundant                                                                   |
| ACK Flag Count              | kept                               |      0.6369 | not redundant                                                                   |
| Bwd IAT Std                 | kept                               |      0.623  | not redundant                                                                   |
| Bwd Packet Length Std       | kept                               |      0.6213 | not redundant                                                                   |
| Init_Win_bytes_backward     | kept                               |      0.6116 | not redundant                                                                   |
| Active Max                  | kept                               |      0.5837 | not redundant                                                                   |
| Active Mean                 | kept                               |      0.5837 | not redundant                                                                   |
| Active Min                  | kept                               |      0.5837 | not redundant                                                                   |
| Idle Max                    | dropped                            |      0.5834 | |r| = 0.980 with kept feature 'Flow IAT Max' (AUC 0.898)                        |
| Idle Mean                   | dropped                            |      0.5834 | |r| = 0.975 with kept feature 'Flow IAT Max' (AUC 0.898)                        |
| Idle Min                    | dropped                            |      0.5834 | |r| = 0.959 with kept feature 'Flow IAT Max' (AUC 0.898)                        |
| Idle Std                    | kept                               |      0.5552 | not redundant                                                                   |
| Active Std                  | kept                               |      0.5551 | not redundant                                                                   |
| URG Flag Count              | kept                               |      0.5509 | not redundant                                                                   |
| Bwd Packet Length Min       | kept                               |      0.5356 | not redundant                                                                   |
| SYN Flag Count              | kept                               |      0.524  | not redundant                                                                   |
| FIN Flag Count              | kept                               |      0.5046 | not redundant                                                                   |
| RST Flag Count              | kept                               |      0.5001 | not redundant                                                                   |
| Destination Port            | with_port set only                 |    nan      | Leakage check: scanner's fixed port list may let a model memorise the lab setup |
| is_well_known_port          | with_port set only (engineered)    |    nan      | Engineered: 1 if destination port < 1024                                        |