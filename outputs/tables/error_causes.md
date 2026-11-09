| error_type     | pattern                                   |   n |   share_of_errors |
|:---------------|:------------------------------------------|----:|------------------:|
| False positive | Benign flow that received no reply at all |   0 |            0      |
| False positive | Benign flow carrying zero payload bytes   |   0 |            0      |
| False positive | Benign flow shorter than 1 ms             |   0 |            0      |
| False negative | Scan flow that did get a reply packet     |  16 |            0.8421 |
| False negative | Scan flow carrying payload bytes          |  10 |            0.5263 |
| False negative | Scan flow aimed at a common service port  |  18 |            0.9474 |
| False negative | Scan flow lasting longer than 1 ms        |  16 |            0.8421 |