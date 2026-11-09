| attack                               | realisation                                                |   recall_under_attack |   evasion_rate |   scan_flows |
|:-------------------------------------|:-----------------------------------------------------------|----------------------:|---------------:|-------------:|
| A0 No attack (clean scan flows)      | Reference: the scan flows exactly as captured              |             0.9986    |       0.001397 |        13604 |
| A1 Timing evasion, 10x slower        | nmap -T2: probes spaced out by an order of magnitude       |             0.771     |       0.229    |        13604 |
| A1 Timing evasion, 100x slower       | nmap -T1: roughly one probe per several seconds            |             0.02551   |       0.9745   |        13604 |
| A1 Timing evasion, 1000x slower      | nmap -T0: a scan paced over hours                          |             7.351e-05 |       0.9999   |        13604 |
| A2 Payload padding, 24 bytes         | nmap --data-length 24: probes stop looking empty           |             0         |       1        |        13604 |
| A2 Payload padding, 64 bytes         | nmap --data-length 64: probes look like small requests     |             0         |       1        |        13604 |
| A3 TCP window reshaping              | Scanner sets an everyday initial window instead of 1024    |             0.7489    |       0.2511   |        13604 |
| A4 Handshake mimicry (connect scan)  | nmap -sT: the connection is completed so a reply exists    |             0         |       1        |        13604 |
| A5 Combined (A1 100x + A2 + A3 + A4) | All four levers used together: the strongest attack tested |             0         |       1        |        13604 |