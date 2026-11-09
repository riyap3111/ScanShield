| error_type     | pattern                                   |   n |   share_of_errors |
|:---------------|:------------------------------------------|----:|------------------:|
| False positive | Benign flow that received no reply at all |   0 |            0      |
| False positive | Benign flow carrying zero payload bytes   |   0 |            0      |
| False positive | Benign flow shorter than 1 ms             |   0 |            0      |
| False negative | Scan flow that did get a reply packet     |  11 |            0.7857 |
| False negative | Scan flow carrying payload bytes          |   7 |            0.5    |
| False negative | Scan flow aimed at a common service port  |  11 |            0.7857 |
| False negative | Scan flow lasting longer than 1 ms        |  13 |            0.9286 |