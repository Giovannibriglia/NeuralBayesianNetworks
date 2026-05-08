## continuous_lg — accuracy (W₁ for continuous / TV for discrete, lower is better)

| n_nodes | gpytorch-gp-predict | nbn-cat-lw | nbn-cat-ve | nbn-flow-lw | nbn-hybrid-router | nbn-lg-lw | nbn-mdn-lw | nbn-neuralcat-lw | pgmpy-bayes-ve | pgmpy-lg-predict | pgmpy-mle-ve | pomegranate-discrete-ve | pyro-empirical-importance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | n/a (cell errored) | n/a (not applicable) | n/a (not applicable) | 0.0564 ± 0.0597 | n/a (not applicable) | 0.0525 ± 0.0502 | 0.0560 ± 0.0549 | n/a (not applicable) | n/a (not applicable) | 0.0465 ± 0.0521 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 50 | n/a (cell errored) | n/a (not applicable) | n/a (not applicable) | 0.1553 ± 0.1726 | n/a (not applicable) | 0.1490 ± 0.1700 | 0.1503 ± 0.1658 | n/a (not applicable) | n/a (not applicable) | 0.1411 ± 0.1712 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 100 | n/a (cell errored) | n/a (not applicable) | n/a (not applicable) | 0.4666 ± 0.4250 | n/a (not applicable) | 0.4696 ± 0.4240 | 0.4795 ± 0.4211 | n/a (not applicable) | n/a (not applicable) | 0.4694 ± 0.4275 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 500 | n/a (cell errored) | n/a (not applicable) | n/a (not applicable) | 0.5969 ± 0.7917 | n/a (not applicable) | 0.5800 ± 0.7856 | 0.5858 ± 0.8005 | n/a (not applicable) | n/a (not applicable) | 0.5842 ± 0.8068 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 1000 | n/a (cell errored) | n/a (not applicable) | n/a (not applicable) | 0.2227 ± 0.0905 | n/a (not applicable) | 0.2216 ± 0.0857 | 0.2158 ± 0.0995 | n/a (not applicable) | n/a (not applicable) | 0.2059 ± 0.0862 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |

## continuous_lg — total_time_s (wall-clock query battery time, seconds)

| n_nodes | gpytorch-gp-predict | nbn-cat-lw | nbn-cat-ve | nbn-flow-lw | nbn-hybrid-router | nbn-lg-lw | nbn-mdn-lw | nbn-neuralcat-lw | pgmpy-bayes-ve | pgmpy-lg-predict | pgmpy-mle-ve | pomegranate-discrete-ve | pyro-empirical-importance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 0.1361 ± 0.1221 | n/a (not applicable) | n/a (not applicable) | 0.0018 ± 0.0003 | n/a (not applicable) | 0.0019 ± 0.0003 | 0.0019 ± 0.0003 | n/a (not applicable) | n/a (not applicable) | 0.0426 ± 0.0033 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 50 | 0.3067 ± 0.1252 | n/a (not applicable) | n/a (not applicable) | 0.0237 ± 0.0004 | n/a (not applicable) | 0.0268 ± 0.0022 | 0.0236 ± 0.0004 | n/a (not applicable) | n/a (not applicable) | 0.2197 ± 0.0300 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 100 | 0.2291 ± 0.0060 | n/a (not applicable) | n/a (not applicable) | 0.0572 ± 0.0025 | n/a (not applicable) | 0.0593 ± 0.0021 | 0.0583 ± 0.0005 | n/a (not applicable) | n/a (not applicable) | 0.8065 ± 0.4931 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 500 | 0.2693 ± 0.0776 | n/a (not applicable) | n/a (not applicable) | 0.3604 ± 0.0032 | n/a (not applicable) | 0.3611 ± 0.0038 | 0.3598 ± 0.0028 | n/a (not applicable) | n/a (not applicable) | 3.3412 ± 1.1975 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 1000 | 0.2364 ± 0.0039 | n/a (not applicable) | n/a (not applicable) | 0.7217 ± 0.0034 | n/a (not applicable) | 0.7216 ± 0.0028 | 0.7215 ± 0.0033 | n/a (not applicable) | n/a (not applicable) | 8.5271 ± 0.4426 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |

## continuous_nongauss — accuracy (W₁ for continuous / TV for discrete, lower is better)

| n_nodes | gpytorch-gp-predict | nbn-cat-lw | nbn-cat-ve | nbn-flow-lw | nbn-hybrid-router | nbn-lg-lw | nbn-mdn-lw | nbn-neuralcat-lw | pgmpy-bayes-ve | pgmpy-lg-predict | pgmpy-mle-ve | pomegranate-discrete-ve | pyro-empirical-importance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | n/a (cell errored) | n/a (not applicable) | n/a (not applicable) | 0.0484 ± 0.0114 | n/a (not applicable) | n/a (not applicable) | 0.0526 ± 0.0192 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 50 | n/a (cell errored) | n/a (not applicable) | n/a (not applicable) | 0.0599 ± 0.0180 | n/a (not applicable) | n/a (not applicable) | 0.0575 ± 0.0151 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 100 | n/a (cell errored) | n/a (not applicable) | n/a (not applicable) | 0.0559 | n/a (not applicable) | n/a (not applicable) | 0.0602 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 500 | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 1000 | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |

## continuous_nongauss — total_time_s (wall-clock query battery time, seconds)

| n_nodes | gpytorch-gp-predict | nbn-cat-lw | nbn-cat-ve | nbn-flow-lw | nbn-hybrid-router | nbn-lg-lw | nbn-mdn-lw | nbn-neuralcat-lw | pgmpy-bayes-ve | pgmpy-lg-predict | pgmpy-mle-ve | pomegranate-discrete-ve | pyro-empirical-importance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 0.1426 ± 0.1293 | n/a (not applicable) | n/a (not applicable) | 0.0116 ± 0.0028 | n/a (not applicable) | n/a (not applicable) | 0.0116 ± 0.0027 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 50 | 0.2433 ± 0.0084 | n/a (not applicable) | n/a (not applicable) | 0.0810 ± 0.0005 | n/a (not applicable) | n/a (not applicable) | 0.0818 ± 0.0017 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 100 | 0.2441 | n/a (not applicable) | n/a (not applicable) | 0.1716 | n/a (not applicable) | n/a (not applicable) | 0.1715 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 500 | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 1000 | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |

## discrete — accuracy (W₁ for continuous / TV for discrete, lower is better)

| n_nodes | gpytorch-gp-predict | nbn-cat-lw | nbn-cat-ve | nbn-flow-lw | nbn-hybrid-router | nbn-lg-lw | nbn-mdn-lw | nbn-neuralcat-lw | pgmpy-bayes-ve | pgmpy-lg-predict | pgmpy-mle-ve | pomegranate-discrete-ve | pyro-empirical-importance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | n/a (not applicable) | 0.0621 ± 0.0191 | 0.0690 ± 0.0153 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | 0.0647 ± 0.0250 | 0.0613 ± 0.0026 | n/a (not applicable) | 0.0532 ± 0.0096 | 0.0596 ± 0.0165 | n/a (cell errored) |
| 50 | n/a (not applicable) | 0.0779 ± 0.0134 | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | 0.0815 ± 0.0066 | 0.0733 ± 0.0133 | n/a (not applicable) | 0.0747 ± 0.0121 | 0.0768 ± 0.0049 | n/a (metric missing) |
| 100 | n/a (not applicable) | 0.0800 ± 0.0107 | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | 0.0826 ± 0.0098 | 0.0760 ± 0.0044 | n/a (not applicable) | 0.0711 ± 0.0049 | 0.0832 ± 0.0222 | n/a (metric missing) |
| 500 | n/a (not applicable) | 0.0890 ± 0.0151 | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | 0.0813 ± 0.0100 | 0.0688 ± 0.0092 | n/a (not applicable) | 0.0868 ± 0.0141 | n/a (metric missing) | n/a (metric missing) |
| 1000 | n/a (not applicable) | 0.0861 ± 0.0112 | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | 0.0815 ± 0.0052 | 0.0777 ± 0.0035 | n/a (not applicable) | 0.0762 ± 0.0035 | n/a (metric missing) | n/a (metric missing) |

## discrete — total_time_s (wall-clock query battery time, seconds)

| n_nodes | gpytorch-gp-predict | nbn-cat-lw | nbn-cat-ve | nbn-flow-lw | nbn-hybrid-router | nbn-lg-lw | nbn-mdn-lw | nbn-neuralcat-lw | pgmpy-bayes-ve | pgmpy-lg-predict | pgmpy-mle-ve | pomegranate-discrete-ve | pyro-empirical-importance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | n/a (not applicable) | 0.0081 ± 0.0019 | 0.0014 ± 0.0004 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | 0.0066 ± 0.0020 | 0.1098 ± 0.0503 | n/a (not applicable) | 0.1076 ± 0.0445 | 1.4212 ± 0.9149 | 94.4566 ± 2.2331 |
| 50 | n/a (not applicable) | 0.0562 ± 0.0031 | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | 0.0553 ± 0.0011 | 0.4428 ± 0.2606 | n/a (not applicable) | 0.4485 ± 0.2617 | 34.6390 ± 5.3932 | n/a (metric missing) |
| 100 | n/a (not applicable) | 0.1206 ± 0.0031 | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | 0.1204 ± 0.0026 | 8.8378 ± 13.7206 | n/a (not applicable) | 8.9458 ± 13.8927 | 100.9742 ± 10.7460 | n/a (metric missing) |
| 500 | n/a (not applicable) | 0.6989 ± 0.0033 | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | 0.7420 ± 0.0090 | 0.8965 ± 0.5631 | n/a (not applicable) | 0.7954 ± 0.6300 | n/a (metric missing) | n/a (metric missing) |
| 1000 | n/a (not applicable) | 1.4158 ± 0.0276 | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | 1.4356 ± 0.0468 | 37.3138 ± 52.4156 | n/a (not applicable) | 36.9763 ± 48.2144 | n/a (metric missing) | n/a (metric missing) |

## hybrid — accuracy (W₁ for continuous / TV for discrete, lower is better)

| n_nodes | gpytorch-gp-predict | nbn-cat-lw | nbn-cat-ve | nbn-flow-lw | nbn-hybrid-router | nbn-lg-lw | nbn-mdn-lw | nbn-neuralcat-lw | pgmpy-bayes-ve | pgmpy-lg-predict | pgmpy-mle-ve | pomegranate-discrete-ve | pyro-empirical-importance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 50 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 100 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 500 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 1000 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |

## hybrid — total_time_s (wall-clock query battery time, seconds)

| n_nodes | gpytorch-gp-predict | nbn-cat-lw | nbn-cat-ve | nbn-flow-lw | nbn-hybrid-router | nbn-lg-lw | nbn-mdn-lw | nbn-neuralcat-lw | pgmpy-bayes-ve | pgmpy-lg-predict | pgmpy-mle-ve | pomegranate-discrete-ve | pyro-empirical-importance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 50 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 100 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 500 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
| 1000 | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (metric missing) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) | n/a (not applicable) |
