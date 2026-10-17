# Round 3: does NBN use a larger budget? — figures and numbers

**R3-1. CPU threads: time (s) at 1, 2, 4, 8 threads and speed-up 1→8 (budget 120 s, 16 GB; same size per row)**

| computation | version | size | 1 | 2 | 4 | 8 | speed-up 1→4 | speed-up 1→8 |
|---|---|---|---|---|---|---|---|---|
| MLE fit (discrete) | this branch | 5000 | 1.51 | 1.48 | 1.4 | 1.49 | 1.08× | 1.02× |
| MLE fit (discrete) | master | 5000 | error | error | error | error | — | — |
| pgmpy MLE fit | baseline | 5000 | 3.97 | 4.14 | 3.95 | 3.99 | 1.00× | 0.99× |
| exact VE (discrete) | this branch | 3 | 10.5 | 6.82 | 4.55 | 3.53 | 2.32× | 2.98× |
| exact VE (discrete) | master | 3 | 10.5 | 6.23 | 4.29 | 3.29 | 2.45× | 3.19× |
| pgmpy VE | baseline | 2 | 0.909 | 1.02 | 0.999 | 1.01 | 0.91× | 0.90× |
| S4 interval sets | this branch | 8 | 1.83 | 1.89 | 1.85 | 1.87 | 0.99× | 0.98× |
| MDN fit (sequential) | master | 200 | 19.2 | 20.4 | 21.5 | 23.8 | 0.89× | 0.81× |
| MDN fit (stacked) | this branch | 500 | 15.1 | 11.8 | 10.8 | 10.3 | 1.40× | 1.47× |
| Pyro SVI (MDN) | baseline | 100 | 13.3 | 14.2 | 14.1 | 16.5 | 0.95× | 0.81× |
| amortized VI training | this branch | 500 | 14 | 8.89 | 6.82 | 5.25 | 2.06× | 2.68× |
| amortized VI training | master | 500 | 24.9 | 25.6 | 30.1 | 36 | 0.83× | 0.69× |
| FlexCode fit | master | 200 | 13.5 | 11.1 | 10.6 | 10.5 | 1.28× | 1.28× |
| FlexCode fit | this branch | 1000 | 18.5 | 16.2 | 15.2 | 15.8 | 1.22× | 1.17× |
| flow fit | master | 100 | 18.3 | 19.4 | 20.7 | 23.3 | 0.88× | 0.78× |
| flow fit | this branch | 200 | 15.4 | 13.8 | 13.2 | 15.2 | 1.17× | 1.02× |
| neural categorical fit | master | 500 | 10.8 | 12.9 | 14.9 | 20 | 0.73× | 0.54× |
| neural categorical fit | this branch | 1000 | 10 | 7.78 | 7.37 | 7.21 | 1.36× | 1.39× |
| Pyro SVI (categorical) | baseline | 500 | 23.9 | 24.1 | 23.8 | 24.8 | 1.00× | 0.96× |
| FlexCode LW queries | this branch | 1000 | 2.43 | 1.88 | 1.64 | 1.58 | 1.49× | 1.54× |
| FlexCode LW queries | master | 50 | 17.4 | 10.9 | 8.62 | 6.74 | 2.01× | 2.58× |
| kNN LW queries | this branch | 100 | 13.9 | 7.92 | 5.22 | 3.72 | 2.67× | 3.75× |
| kNN LW queries | master | 10 | 21.4 | 12 | 9.48 | 6.92 | 2.26× | 3.10× |
| LW queries (non-Gaussian) | this branch | 5000 | 1.85 | 1.67 | 1.61 | 1.69 | 1.14× | 1.09× |
| Pyro vectorised LW | baseline | 1000 | 11.8 | 11.5 | 11 | 11.7 | 1.07× | 1.01× |
| LW queries (discrete) | this branch | 1000 | 11.3 | 9.63 | 9.67 | 10.7 | 1.17× | 1.05× |
| LW queries (discrete) | master | 1000 | 84.8 | 58.6 | 53.3 | 49.7 | 1.59× | 1.71× |
| certified queries | this branch | 500 | 15.7 | 12.1 | 10.7 | 11 | 1.46× | 1.42× |
| amortized IS | this branch | 1000 | 1.2 | 1.53 | 1.83 | 1.52 | 0.65× | 0.79× |
| amortized IS | master | 1000 | 9.78 | 7.7 | 7.22 | 7.55 | 1.35× | 1.30× |

**R3-2. GPU (Apple MPS) vs CPU: time (s); speed-up = CPU time / GPU time. * exact VE: the guard answers a different number of queries per device, so times are not comparable**

| computation | size | CPU 1 thread | CPU best (threads) | GPU | GPU vs 1 thread | GPU vs best CPU |
|---|---|---|---|---|---|---|
| MLE fit (discrete) | 5000 | 1.51 | 1.4 (4) | 11.6 | 0.13× | 0.12× |
| S4 interval sets | 8 | 1.83 | 1.83 (1) | 2.21 | 0.83× | 0.83× |
| exact VE (discrete) * | 3 | 10.5 (9/10) | 3.53 (8) | 6.41 (6/10) | — | — |
| FlexCode fit | 1000 | 18.5 | 15.2 (4) | 11.1 | 1.66× | 1.37× |
| MDN fit (stacked) | 500 | 15.1 | 10.3 (8) | 7.46 | 2.03× | 1.38× |
| Pyro SVI (MDN) | 100 | 13.3 | 13.3 (1) | timeout | — | — |
| Pyro SVI (categorical) | 500 | 23.9 | 23.8 (4) | timeout | — | — |
| amortized VI training | 500 | 14 | 5.25 (8) | 3.9 | 3.60× | 1.35× |
| flow fit | 200 | 15.4 | 13.2 (4) | 13.8 | 1.12× | 0.96× |
| neural categorical fit | 1000 | 10 | 7.21 (8) | 5.24 | 1.91× | 1.38× |
| FlexCode LW queries | 1000 | 2.43 | 1.58 (8) | 0.9 | 2.70× | 1.75× |
| kNN LW queries | 100 | 13.9 | 3.72 (8) | 14 | 1.00× | 0.27× |
| LW queries (discrete) | 1000 | 11.3 | 9.63 (2) | 8.03 | 1.41× | 1.20× |
| LW queries (non-Gaussian) | 5000 | 1.85 | 1.61 (4) | 2.94 | 0.63× | 0.55× |
| Pyro vectorised LW | 1000 | 11.8 | 11 (4) | 30 | 0.39× | 0.37× |
| amortized IS | 1000 | 1.2 | 1.2 (1) | 3.79 | 0.32× | 0.32× |
| certified queries | 500 | 15.7 | 10.7 (4) | 4.9 | 3.20× | 2.19× |

**R3-3. Largest graph (nodes) finished: reference budget (1 thread, 4 GB, 30 s) vs large budget (8 threads, 16 GB, 120 s)**

| computation | reference: largest | stopped by | large: largest | stopped by | gain in nodes |
|---|---|---|---|---|---|
| exact VE, loose ER graphs | 100 | ok at 200 | 100 | ok at 200 | 1× |
| pgmpy VE, loose ER graphs | 100 | oom at 200 | 100 | crash at 200 | 1× |
| LW queries, ER | 5e+03 | none (max tested) | 1e+04 | none (max tested) | 2× |
| LW queries, local graphs | 5e+03 | none (max tested) | 1e+04 | none (max tested) | 2× |
| certified queries, local | 1e+03 | timeout at 2000 | 1e+04 | none (max tested) | 10× |
| MDN fit (stacked) | 500 | timeout at 1000 | 5e+03 | timeout at 10000 | 10× |
| Pyro SVI (MDN) | 200 | timeout at 500 | 500 | timeout at 1000 | 2× |
| neural categorical fit | 1e+03 | none (max tested) | 5e+03 | none (max tested) | 5× |
| Pyro SVI (categorical) | 500 | timeout at 1000 | 2e+03 | timeout at 5000 | 4× |
| FlexCode fit | 1e+03 | none (max tested) | 5e+03 | none (max tested) | 5× |
| flow fit | 200 | timeout at 500 | 1e+03 | timeout at 2000 | 5× |
| kNN LW queries | 200 | timeout at 500 | 2e+03 | timeout at 5000 | 10× |
| FlexCode LW queries | 1e+03 | none (max tested) | 5e+03 | none (max tested) | 5× |
| amortized IS | 1e+03 | none (max tested) | 5e+03 | none (max tested) | 5× |
| amortized VI (10 epochs) | 1e+03 | none (max tested) | 5e+03 | none (max tested) | 5× |

**R3-4. Exact inference on denser graphs (100 nodes, 10 queries): queries answered and the largest elimination width answered**

| avg degree | budget | NBN answered | NBN max width answered | NBN time (s) | pgmpy |
|---|---|---|---|---|---|
| 1 | 1 thread, 4 GB, 30 s | 10/10 | 5 | 0.0108 | 10/10 (0.00925 s) |
| 1.5 | 1 thread, 4 GB, 30 s | 10/10 | 6 | 0.0132 | 10/10 (0.0111 s) |
| 2 | 1 thread, 4 GB, 30 s | 10/10 | 15 | 0.268 | 10/10 (0.927 s) |
| 3 | 1 thread, 4 GB, 30 s | 8/10 | 16 | 5.03 | timeout |
| 4 | 1 thread, 4 GB, 30 s | 1/10 | 16 | 2.53 | — |
| 4.5 | 1 thread, 4 GB, 30 s | 1/10 | 16 | 2.8 | oom |
| 5 | 1 thread, 4 GB, 30 s | 1/10 | 12 | 0.379 | — |
| 5.5 | 1 thread, 4 GB, 30 s | 0/10 | — | oom | — |
| 6 | 1 thread, 4 GB, 30 s | 0/10 | — | oom | — |
| 2 | 8 threads, 16 GB, 120 s | 10/10 | 15 | 0.145 | 10/10 (1.01 s) |
| 3 | 8 threads, 16 GB, 120 s | 9/10 | 17 | 3.51 | oom |
| 4 | 8 threads, 16 GB, 120 s | 1/10 | 16 | 0.875 | — |
| 4.5 | 8 threads, 16 GB, 120 s | 1/10 | 16 | 1.01 | — |
| 5 | 8 threads, 16 GB, 120 s | 2/10 | 17 | 1.8 | — |
| 5.5 | 8 threads, 16 GB, 120 s | 0/10 | — | oom | — |

**R3-5. Likelihood weighting with more particles: time (s) for 10 queries at 1 and 8 threads, and mean max-abs error vs exact VE (discrete only)**

| network | particles | 1 thread | 8 threads | speed-up | error vs exact | ESS/N |
|---|---|---|---|---|---|---|
| discrete, local, 1,000 nodes | 100,000 | 12.1 | 10.6 | 1.14× | 0.0039 | 0.314 |
| discrete, local, 1,000 nodes | 300,000 | 36.1 | 28.7 | 1.26× | 0.0019 | 0.316 |
| discrete, local, 1,000 nodes | 1,000,000 | 126 | 100 | 1.26× | 0.0010 | 0.315 |
| non-Gaussian MDN, ER, 5,000 nodes | 10,000 | 1.83 | 1.63 | 1.12× | — | 0.164 |
| non-Gaussian MDN, ER, 5,000 nodes | 100,000 | 16.6 | 16.2 | 1.03× | — | 0.167 |
| non-Gaussian MDN, ER, 5,000 nodes | 300,000 | 51.6 | 49.8 | 1.04× | — | 0.167 |
