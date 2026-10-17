# Round 2: other mechanisms and engines — figures and numbers

**R2-0. Speed-ups of this branch at the largest size both versions finished**

| what | compared with | speed-up |
|---|---|---|
| Neural categorical fit (1000 nodes) | master | 2.2× |
| Neural categorical fit (500 nodes) | Pyro | 4.6× |
| FlexCode fit (200 nodes) | master | 3.3× |
| Flow fit (100 nodes) | master | 2.4× |
| kNN LW queries (10 nodes) | master | 3.5× |
| kNN LW queries (50 nodes) | first branch | 2.9× |
| FlexCode LW queries (50 nodes) | master | 27.4× |
| FlexCode LW queries (1000 nodes) | first branch | 6.8× |
| Amortized IS queries (1000 nodes) | master | 6.6× |
| Amortized IS queries (1000 nodes) | first branch | 2.0× |
| Amortized VI step (500 nodes) | master | 1.7× |

*Where the older version timed out at every larger size, the gap is larger than shown.*

**R2-1. Neural-mechanism fitting time (s) and fit quality (10 epochs, 10 000 rows, 1 thread)**

| mechanism | nodes | NBN before (sequential) | NBN stacked | Pyro SVI | speed-up vs before | LL before | LL stacked | LL Pyro |
|---|---|---|---|---|---|---|---|---|
| ncat | 50 | 1.4 | 0.839 | 2.35 | 1.7× | -0.924 | -0.924 | -0.954 |
| ncat | 200 | 4.67 | 2.02 | 9.64 | 2.3× | -0.962 | -0.962 | -0.976 |
| ncat | 500 | 11.2 | 5.18 | 23.7 | 2.2× | -0.957 | -0.958 | -0.976 |
| flex | 50 | 3.66 | 1.3 | n/a | 2.8× | -1.45 | -1.46 | n/a |
| flex | 200 | 13.8 | 4.25 | n/a | 3.3× | -1.51 | -1.51 | n/a |
| flex | 500 | timeout | 10.3 | n/a | — | timeout | -1.5 | n/a |
| flow | 50 | 9.52 | 3.96 | n/a | 2.4× | -1.24 | -1.23 | n/a |
| flow | 200 | timeout | 16.1 | n/a | — | timeout | -1.26 | n/a |
| flow | 500 | — | timeout | n/a | — | — | timeout | n/a |

*LL = mean test log-likelihood per node and row (higher is better). 'Before' is the sequential path, identical on master and the first branch.*

**R2-2. Ten likelihood-weighting queries (10k samples) on networks of kNN or FlexCode mechanisms (s)**

| mechanism | nodes | master | first branch | this branch | vs master | vs first branch |
|---|---|---|---|---|---|---|
| knn | 10 | 20.3 | 13.3 | 5.89 | 3.5× | 2.3× |
| knn | 20 | timeout | 15.8 | 6.49 | — | 2.4× |
| knn | 50 | — | 30 | 10.3 | — | 2.9× |
| knn | 100 | — | timeout | 14 | — | — |
| knn | 200 | — | — | 17.9 | — | — |
| flex | 10 | 3.26 | 2 | 0.247 | 13.2× | 8.1× |
| flex | 50 | 17.2 | 4.62 | 0.629 | 27.4× | 7.3× |
| flex | 100 | timeout | 5.99 | 0.825 | — | 7.3× |
| flex | 500 | — | 14.3 | 2.08 | — | 6.9× |
| flex | 1000 | — | 16 | 2.35 | — | 6.8× |

**R2-3. Amortized IS on loose graphs: time for 10 queries (s), median effective sample size, median error vs exact**

| nodes | time master | time this branch | speed-up | ESS/N master | ESS/N this branch | error master | error this branch |
|---|---|---|---|---|---|---|---|
| 50 | 0.921 | 0.714 | 1.3× | 0.039 | 0.0667 | 0.0244 | 0.0154 |
| 200 | 1.55 | 0.332 | 4.7× | 0.228 | 0.229 | 0.0116 | 0.0113 |
| 500 | 3.92 | 0.764 | 5.1× | 0.275 | 0.272 | 0.0113 | 0.0113 |
| 1000 | 8.69 | 1.32 | 6.6× | 0.322 | 0.319 | 0.008 | 0.012 |

**R2-3b. Amortized IS, same estimator: mean absolute error vs exact VE and median ESS/N across seeds (10 queries per seed)**

| nodes | version | seeds | mean error | sd across seeds | median ESS/N | mean time (s) |
|---|---|---|---|---|---|---|
| 200 | master | 6 | 0.0299 | 0.0298 | 0.204 | 2.44 |
| 200 | this branch | 6 | 0.0151 | 0.00676 | 0.233 | 0.709 |
| 1000 | master | 6 | 0.00871 | 0.00122 | 0.271 | 8.22 |
| 1000 | this branch | 6 | 0.00985 | 0.00257 | 0.271 | 1.06 |

**R2-4. Amortized VI: time per ELBO training step (ms)**

| nodes | master | first branch | this branch | speed-up vs master |
|---|---|---|---|---|
| 50 | 49.9 | 50.1 | 38.1 | 1.3× |
| 200 | 211 | 211 | 143 | 1.5× |
| 500 | 572 | 582 | 341 | 1.7× |
| 1000 | timeout | timeout | 665 | — |

**R2-5. kNN neighbour search, 10 000 queries (s, 1 thread)**

| training rows | parent dims | old (float32, ties unspecified) | new brute (exact) | new k-d tree (exact) | discrete parents old | discrete new |
|---|---|---|---|---|---|---|
| 2000 | 1 | 0.0996 | 0.122 | 0.0209 | 0.0289 | 0.00239 |
| 10000 | 1 | 0.233 | 0.294 | 0.0463 | 0.147 | 0.00318 |
| 50000 | 1 | 0.978 | 1.14 | 0.122 | 0.699 | 0.00579 |
| 2000 | 2 | 0.122 | 0.146 | 0.0323 | 0.0337 | 0.00313 |
| 10000 | 2 | 0.257 | 0.338 | 0.0713 | 0.167 | 0.00525 |
| 50000 | 2 | 1.01 | 1.21 | 0.168 | 0.735 | 0.0144 |
| 2000 | 4 | 0.112 | 0.137 | 0.0553 | 0.107 | 0.0152 |
| 10000 | 4 | 0.261 | 0.343 | 0.13 | 0.254 | 0.0537 |
| 50000 | 4 | 1.01 | 1.22 | 0.286 | 0.969 | 0.258 |
| 2000 | 8 | 0.114 | 0.142 | 0.12 | 0.104 | 0.134 |
| 10000 | 8 | 0.262 | 0.352 | 0.454 | 0.264 | 0.353 |
| 50000 | 8 | 1.02 | 1.25 | 1.21 | 1.02 | 1.24 |

**R2-6. FlexCode grid density (normaliser), seconds**

| rows | before (broadcast) | after (matmul) | speed-up |
|---|---|---|---|
| 10000 | 0.035 | 0.00122 | 29× |
| 100000 | 0.38 | 0.014 | 27× |
