# Round 4: a larger budget buys a better answer — figures and numbers

**R4-1. Exact queries answered (of 10) on denser graphs (ER, 100 nodes, K = 3): plain VE vs conditioning within 20 s per query.  Not finished: conditioning started (at most 10^6 instantiations) but stopped at 20 s, with the time it would need projected from its measured rate; over cap: more than 10^6 instantiations, not attempted**

| config | memory | avg degree | plain VE | conditioning <= 20 s | median s (answered) | cutset |C| (median / max) | not finished: n, projected s (median / max) | over cap |
|---|---|---|---|---|---|---|---|---|
| CPU, 1 thread | 1 GB | 3 | 6/10 | 8/10 | 2.54 | 0 / 4 | 2: 66.1 / 108 | 0 |
| CPU, 1 thread | 1 GB | 4.5 | 0/10 | 1/10 | 9.27 | 13 / 18 | 3: 1.1e+05 / 3e+05 | 6 |
| CPU, 1 thread | 1 GB | 6 | 0/10 | 0/10 | — | 18 / 21 | 2: 4.69e+05 / 9.27e+05 | 8 |
| CPU, 1 thread | 4 GB | 3 | 8/10 | 9/10 | 3.14 | 0 / 2 | 1: 110 / 110 | 0 |
| CPU, 1 thread | 4 GB | 4.5 | 1/10 | 1/10 | 10.5 | 11 / 16 | 6: 1.72e+06 / 8.35e+06 | 3 |
| CPU, 1 thread | 4 GB | 6 | 0/10 | 0/10 | — | 16 / 19 | 2: 4.4e+05 / 8.72e+05 | 8 |
| CPU, 1 thread | 16 GB | 3 | 9/10 | 9/10 | 3.08 | 0 / 1 | 1: 177 / 177 | 0 |
| CPU, 1 thread | 16 GB | 4.5 | 1/10 | 1/10 | 10.5 | 10 / 15 | 7: 3.51e+06 / 3.98e+07 | 2 |
| CPU, 1 thread | 16 GB | 6 | 0/10 | 0/10 | — | 15 / 18 | 3: 8.58e+05 / 2.1e+07 | 7 |
| CPU, 20 threads | 1 GB | 3 | 6/10 | 9/10 | 0.535 | 0 / 4 | 1: 20.7 / 20.7 | 0 |
| CPU, 20 threads | 1 GB | 4.5 | 0/10 | 1/10 | 1.52 | 13 / 18 | 3: 1.86e+04 / 5.61e+04 | 6 |
| CPU, 20 threads | 1 GB | 6 | 0/10 | 0/10 | — | 18 / 21 | 2: 9.87e+04 / 1.95e+05 | 8 |
| CPU, 20 threads | 4 GB | 3 | 8/10 | 10/10 | 0.568 | 0 / 2 | 0 | 0 |
| CPU, 20 threads | 4 GB | 4.5 | 1/10 | 1/10 | 1.51 | 11 / 16 | 6: 2.64e+05 / 1.6e+06 | 3 |
| CPU, 20 threads | 4 GB | 6 | 0/10 | 0/10 | — | 16 / 19 | 2: 9.38e+04 / 1.86e+05 | 8 |
| CPU, 20 threads | 16 GB | 3 | 9/10 | 10/10 | 0.571 | 0 / 1 | 0 | 0 |
| CPU, 20 threads | 16 GB | 4.5 | 1/10 | 1/10 | 1.51 | 10 / 15 | 7: 5.1e+05 / 6.27e+06 | 2 |
| CPU, 20 threads | 16 GB | 6 | 0/10 | 0/10 | — | 15 / 18 | 3: 1.75e+05 / 3.9e+06 | 7 |
| CUDA (TITAN X) | 1 GB | 3 | 6/10 | 10/10 | 0.0588 | 0 / 4 | 0 | 0 |
| CUDA (TITAN X) | 1 GB | 4.5 | 0/10 | 2/10 | 2.93 | 13 / 18 | 2: 2.61e+03 / 4.07e+03 | 6 |
| CUDA (TITAN X) | 1 GB | 6 | 0/10 | 0/10 | — | 18 / 21 | 2: 8.19e+03 / 1.63e+04 | 8 |
| CUDA (TITAN X) | 4 GB | 3 | 8/10 | 10/10 | 0.061 | 0 / 3 | 0 | 0 |
| CUDA (TITAN X) | 4 GB | 4.5 | 1/10 | 2/10 | 2.64 | 12 / 17 | 4: 2.99e+04 / 5.74e+04 | 4 |
| CUDA (TITAN X) | 4 GB | 6 | 0/10 | 0/10 | — | 17 / 20 | 2: 7.81e+03 / 1.55e+04 | 8 |
| CUDA (TITAN X) | 10 GB | 3 | 9/10 | 10/10 | 0.0614 | 0 / 2 | 0 | 0 |
| CUDA (TITAN X) | 10 GB | 4.5 | 1/10 | 2/10 | 2.52 | 11 / 16 | 5: 3.04e+04 / 1.12e+05 | 3 |
| CUDA (TITAN X) | 10 GB | 6 | 0/10 | 0/10 | — | 16 / 19 | 2: 7.63e+03 / 1.52e+04 | 8 |
| Apple MPS | 1 GB | 3 | 0/10 | 9/10 | 2.03 | 3.5 / 8 | 1: 43 / 43 | 0 |
| Apple MPS | 1 GB | 4.5 | 0/10 | 1/10 | 4.82 | 16.5 / 21 | 1: 174 / 174 | 8 |
| Apple MPS | 1 GB | 6 | 0/10 | 0/10 | — | 21 / 24 | 1: 2.53e+03 / 2.53e+03 | 9 |
| Apple MPS | 4 GB | 3 | 6/10 | 9/10 | 1.27 | 1 / 6 | 1: 188 / 188 | 0 |
| Apple MPS | 4 GB | 4.5 | 0/10 | 1/10 | 5.24 | 14 / 19 | 3: 5.69e+04 / 8.35e+04 | 6 |
| Apple MPS | 4 GB | 6 | 0/10 | 0/10 | — | 19 / 22 | 1: 2.21e+03 / 2.21e+03 | 9 |
| Apple MPS | 16 GB | 3 | 6/10 | 9/10 | 1.26 | 1 / 6 | 1: 186 / 186 | 0 |
| Apple MPS | 16 GB | 4.5 | 0/10 | 1/10 | 5.9 | 14 / 19 | 3: 5.63e+04 / 8.26e+04 | 6 |
| Apple MPS | 16 GB | 6 | 0/10 | 0/10 | — | 19 / 22 | 1: 2.21e+03 / 2.21e+03 | 9 |

**R4-2. Certified sampling, anytime mode: median certified width (95%, all classes) by time budget; coverage of the exact answer where it is known**

| problem | config | mode | 0.25 s | 1 s | 4 s | 16 s | covers exact |
|---|---|---|---|---|---|---|---|
| er n=100 deg 3 | CPU, 1 thread | plain | 0.148 | 0.0596 | 0.0255 | 0.017 | 32/32 |
| er n=100 deg 3 | CPU, 1 thread | collapsed | 0.61 | 0.131 | 0.0494 | 0.023 | 32/32 |
| er n=100 deg 3 | CPU, 20 threads | plain | 0.195 | 0.039 | 0.0195 | 0.00878 | 32/32 |
| er n=100 deg 3 | CPU, 20 threads | collapsed | 0.512 | 0.0665 | 0.0265 | 0.00963 | 32/32 |
| er n=100 deg 3 | CUDA (TITAN X) | plain | 0.0381 | 0.0115 | 0.00548 | 0.00269 | 32/32 |
| er n=100 deg 3 | CUDA (TITAN X) | collapsed | 0.092 | 0.01 | 0.0038 | 0.00167 | 32/32 |
| er n=100 deg 3 | Apple MPS | plain | 0.0374 | 0.0169 | 0.0079 | 0.00383 | 24/24 |
| er n=100 deg 3 | Apple MPS | collapsed | 0.0605 | 0.0215 | 0.00894 | 0.00357 | 24/24 |
| er n=100 deg 4.5 | CPU, 1 thread | plain | 0.333 | 0.0691 | 0.0365 | 0.0229 | 4/4 |
| er n=100 deg 4.5 | CPU, 1 thread | collapsed | 0.795 | 0.795 | 0.14 | 0.0605 | 4/4 |
| er n=100 deg 4.5 | CPU, 20 threads | plain | 0.333 | 0.0691 | 0.0273 | 0.0118 | 4/4 |
| er n=100 deg 4.5 | CPU, 20 threads | collapsed | 0.795 | 0.764 | 0.0907 | 0.0251 | 4/4 |
| er n=100 deg 4.5 | CUDA (TITAN X) | plain | 0.114 | 0.0178 | 0.00796 | 0.00376 | 4/4 |
| er n=100 deg 4.5 | CUDA (TITAN X) | collapsed | 0.796 | 0.524 | 0.01 | 0.00361 | 4/4 |
| er n=100 deg 4.5 | Apple MPS | plain | 0.0589 | 0.0273 | 0.0118 | 0.00545 | — |
| er n=100 deg 4.5 | Apple MPS | collapsed | 0.799 | 0.0532 | 0.0165 | 0.00615 | — |
| local n=1000 deg 2 | CPU, 1 thread | plain | 0.448 | 0.198 | 0.0662 | 0.0375 | 40/40 |
| local n=1000 deg 2 | CPU, 1 thread | collapsed | 2e-06 | 2e-06 | 2e-06 | 2e-06 | 40/40 |
| local n=1000 deg 2 | CPU, 20 threads | plain | 0.448 | 0.239 | 0.0763 | 0.0358 | 40/40 |
| local n=1000 deg 2 | CPU, 20 threads | collapsed | 2e-06 | 2e-06 | 2e-06 | 2e-06 | 40/40 |
| local n=1000 deg 2 | CUDA (TITAN X) | plain | 0.434 | 0.128 | 0.026 | 0.0119 | 40/40 |
| local n=1000 deg 2 | CUDA (TITAN X) | collapsed | 2e-06 | 2e-06 | 2e-06 | 2e-06 | 40/40 |
| local n=1000 deg 2 | Apple MPS | plain | 0.312 | 0.0575 | 0.0253 | 0.012 | 40/40 |
| local n=1000 deg 2 | Apple MPS | collapsed | 0.0858 | 0.0494 | 0.0255 | 0.00836 | 40/40 |

**R4-3. Bounded conditioning (sound bounds, no probability): median width by time budget; skew = factor on the CPT logits**

| config | skew | 1 s | 4 s | 16 s | 64 s | exact at largest | median prior mass covered |
|---|---|---|---|---|---|---|---|
| CPU, 1 thread | 1 | 1 | 1 | 1 | 1 | 1/5 | 0.000464 |
| CPU, 1 thread | 3 | 1 | 1 | 1 | 1 | 1/5 | 0.0518 |
| CPU, 20 threads | 1 | 1 | 1 | 1 | 1 | 1/5 | 0.00153 |
| CPU, 20 threads | 3 | 1 | 1 | 1 | 1 | 1/5 | 0.148 |
| CUDA (TITAN X) | 1 | 1 | 1 | 1 | 1 | 1/5 | 0.00613 |
| CUDA (TITAN X) | 3 | 1 | 1 | 0.999 | 0.998 | 1/5 | 0.523 |
| Apple MPS | 1 | 1 | 1 | 1 | 1 | 1/5 | 0.000116 |
| Apple MPS | 3 | 1 | 1 | 1 | 0.999 | 1/5 | 0.248 |
