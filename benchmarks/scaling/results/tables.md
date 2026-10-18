## Frontier: largest size solved within the budget (per seed)

- **discrete_infer (local)**: nbn_cert: [1000, 1000, 5000]; nbn_lw: [5000, 5000, 5000]; nbn_lw@master: [200]; nbn_ve: [5000, 5000, 5000]; nbn_ve@master: [5000]; pgmpy_ve: [5000, 5000, 5000]

- **discrete_infer (er)**: nbn_cert: [5000, 5000, 5000]; nbn_lw: [5000, 5000, 5000]; nbn_lw@master: [200]; nbn_ve: [100, 100, 200]; nbn_ve@master: [100, 100, 100]; pgmpy_ve: [100, 100, 100]

- **discrete_fit**: nbn: [5000]; pgmpy: [5000]

- **discrete_sets**: nbn_interval: [5000]; nbn_intervals_batched: [5000]; nbn_l1: [5000]

- **cont_fit**: nbn_local: [200]; nbn_local[mps]: [20]; nbn_stacked: [500]; nbn_stacked[mps]: [1000]; pyro_svi: [200]; pyro_svi[mps]: [10]

- **cont_infer**: nbn_lw: [5000]; nbn_lw@master: [2000]; pyro_is: [0]; pyro_vec: [2000]

### discrete_fit, x = n

| nodes | nbn | pgmpy |
|---|---|---|
| 10 | 0.00364 | 0.0128 |
| 20 | 0.00659 | 0.0163 |
| 50 | 0.0152 | 0.0416 |
| 100 | 0.0297 | 0.0869 |
| 200 | 0.0596 | 0.16 |
| 500 | 0.142 | 0.367 |
| 1000 | 0.278 | 0.741 |
| 2000 | 0.575 | 1.49 |
| 5000 | 1.48 | 3.95 |

### discrete_fit, x = deg

| mean in-degree (edges) | nbn | pgmpy |
|---|---|---|
| 1 (224) | 0.0525 | 0.12 |
| 1.5 (313) | 0.0556 | 0.14 |
| 2 (416) | 0.0602 | 0.162 |
| 3 (606) | 0.0657 | 0.196 |
| 4 (795) | 0.0714 | 0.33 |
| 6 (1103) | 0.0869 | 0.617 |
| 8 (1351) | 0.102 | 0.922 |

### discrete_sets, x = n

| nodes | nbn_interval | nbn_intervals_batched | nbn_l1 |
|---|---|---|---|
| 10 | 0.0122 | 0.00258 | 0.00125 |
| 20 | 0.0236 | 0.00388 | 0.00218 |
| 50 | 0.0651 | 0.0117 | 0.00496 |
| 100 | 0.133 | 0.0367 | 0.00979 |
| 200 | 0.259 | 0.0506 | 0.0181 |
| 500 | 0.586 | 0.0962 | 0.0422 |
| 1000 | 1.24 | 0.2 | 0.0883 |
| 2000 | 2.38 | 0.428 | 0.175 |
| 5000 | 6.27 | 1.04 | 0.429 |

### discrete_sets, x = deg

| mean in-degree (edges) | nbn_interval | nbn_intervals_batched | nbn_l1 |
|---|---|---|---|
| 1 (224) | 0.229 | 0.019 | 0.0146 |
| 1.5 (313) | 0.235 | 0.025 | 0.0158 |
| 2 (416) | 0.259 | 0.0514 | 0.0186 |
| 3 (606) | 0.287 | 0.101 | 0.0229 |
| 4 (795) | 0.525 | 0.355 | 0.0299 |
| 6 (1103) | 1.16 | 1.05 | 0.046 |
| 8 (1351) | 1.86 | 1.81 | 0.0595 |

### cont_fit, x = n

| nodes | nbn_local | nbn_local[mps] | nbn_stacked | nbn_stacked[mps] | pyro_svi | pyro_svi[mps] |
|---|---|---|---|---|---|---|
| 10 | 1.24 | 7.91 | 0.624 | 0.809 | 1.51 | 17.1 |
| 20 | 2.11 | 17.3 | 0.979 | 0.848 | 2.7 | timeout |
| 50 | 4.85 | timeout | 1.69 | 1.17 | 6.36 |  |
| 100 | 9.65 |  | 3.15 | 1.78 | 12.8 |  |
| 200 | 19 |  | 5.62 | 3.24 | 27.5 |  |
| 500 | timeout |  | 15.1 | 7.67 | timeout |  |
| 1000 |  |  | timeout | 15.1 |  |  |
| 2000 |  |  |  | timeout |  |  |

### cont_fit, x = deg

| mean in-degree (edges) | nbn_local | nbn_local[mps] | nbn_stacked | nbn_stacked[mps] | pyro_svi | pyro_svi[mps] |
|---|---|---|---|---|---|---|
| 1 (112) | 8.1 | timeout | 2.55 | 1.59 | 12.4 | timeout |
| 1.5 (156) | 9.17 |  | 2.84 | 1.71 | 12.6 |  |
| 2 (211) | 9.55 |  | 3.14 | 1.81 | 12.8 |  |
| 3 (309) | 10.3 |  | 3.39 | 1.86 | 13.1 |  |
| 4 (418) | 10.5 |  | 3.46 | 1.92 | 13.6 |  |
| 6 (566) | 10.8 |  | 3.54 | 1.95 | 13.8 |  |
| 8 (664) | 10.8 |  | 3.58 | 1.94 | 13.2 |  |

### cont_infer, x = n

| nodes | nbn_lw | nbn_lw@master | pyro_is | pyro_vec |
|---|---|---|---|---|
| 10 | 0.05 | 0.0874 | timeout | 0.101 |
| 20 | 0.0662 | 0.19 |  | 0.216 |
| 50 | 0.128 | 0.509 |  | 0.559 |
| 100 | 0.169 | 1.01 |  | 1.13 |
| 200 | 0.218 | 2.03 |  | 2.28 |
| 500 | 0.406 | 4.98 |  | 5.68 |
| 1000 | 0.49 | 10.1 |  | 11.6 |
| 2000 | 0.643 | 20.5 |  | 23.2 |
| 5000 | 1.8 | timeout |  | oom |

### cont_infer, x = deg

| mean in-degree (edges) | nbn_lw | nbn_lw@master | pyro_is | pyro_vec |
|---|---|---|---|---|
| 1 (112) | 0.0733 | 0.931 | timeout | 1.07 |
| 1.5 (156) | 0.108 | 0.926 |  | 1.11 |
| 2 (211) | 0.169 | 1.02 |  | 1.13 |
| 3 (309) | 0.232 | 1 |  | 1.17 |
| 4 (418) | 0.313 | 1.09 |  | 1.19 |
| 6 (566) | 0.387 | 1.1 |  | 1.19 |
| 8 (664) | 0.386 | 1.05 |  | 1.2 |

### discrete_infer (local), x = n

| nodes | nbn_cert | nbn_lw | nbn_lw@master | nbn_ve | nbn_ve@master | pgmpy_ve |
|---|---|---|---|---|---|---|
| 10 | 0.493 | 0.241 | 0.492 | 0.00674 | 0.00685 | 0.00359 |
| 20 | 0.992 | 0.477 | 1.25 | 0.0105 | 0.0117 | 0.00665 |
| 50 | 2.15 | 0.971 | 3.69 | 0.0334 | 0.0485 | 0.0141 |
| 100 | 4.2 | 1.96 | 7.75 | 0.0302 | 0.1 | 0.0269 |
| 200 | 7.56 | 3.45 | 15.8 | 0.063 | 0.363 | 0.0616 |
| 500 | 13.8 | 7.25 | timeout | 0.0972 | 0.565 | 0.107 |
| 1000 | 16.8 | 8.48 |  | 0.0714 | 3.66 | 0.125 |
| 2000 | 18.8 (timeout in 2/3) | 15.6 |  | 0.186 | 6.08 | 0.388 |
| 5000 | 29.9 | 15.8 |  | 0.167 | 2.42 | 1.36 |

### discrete_infer (er), x = n

| nodes | nbn_cert | nbn_lw | nbn_lw@master | nbn_ve | nbn_ve@master | pgmpy_ve |
|---|---|---|---|---|---|---|
| 10 | 0.474 | 0.22 | 0.499 | 0.00603 | 0.00641 | 0.0034 |
| 20 | 0.858 | 0.379 | 1.33 | 0.00823 | 0.0101 | 0.00558 |
| 50 | 1.5 | 0.629 | 4.02 | 0.0292 | 0.0319 | 0.0099 |
| 100 | 2.39 | 1.04 | 8.79 | 0.0243 | 0.0339 | 0.0361 |
| 200 | 3.66 | 1.55 | 17.8 | 0.819 (27/30) | 0.908 (26/30) | oom (3/3) |
| 500 | 5.54 | 2.78 | timeout | 0.631 (14/30) | 0.824 (14/30) |  |
| 1000 | 6.49 | 3.25 |  | 0.999 (9/30) | 1.29 (9/30) |  |
| 2000 | 9.18 | 4.69 |  | 0.985 (7/30) | 1.73 (7/30) |  |
| 5000 | 14.9 | 7.75 |  | 4.39 (7/30) | 7.2 (7/30) |  |

### discrete_infer (er), x = deg

| mean in-degree (edges) | nbn_cert | nbn_lw | nbn_lw@master | nbn_ve | nbn_ve@master | pgmpy_ve |
|---|---|---|---|---|---|---|
| 1 (107) | 1.29 | 0.588 | 7.66 | 0.0098 | 0.013 | 0.00834 |
| 1.5 (146) | 1.84 | 0.821 | 7.78 | 0.0132 | 0.0163 | 0.0111 |
| 2 (201) | 2.3 | 1.01 | 8.04 | 0.0248 | 0.421 | 0.0383 |
| 3 (303) | 2.86 | 1.19 | 8.48 | 2.73 (21/30) | 4 (4/10) | timeout (3/3) |
| 4 (415) | 3.33 | 1.3 | 8.82 | 0.238 (3/30) | oom |  |
| 6 (563) | 4.97 | 1.91 | 9.39 | oom (3/3) |  |  |
| 8 (665) | 5.96 | 2.24 | 9.7 |  |  |  |

Width bounds on 690 exact-VE queries: median gap (min-fill width − lower bound) = 0; 122 refused by the memory guard, of which 9 are *proven* infeasible by the lower bound (needs lb > 17.7).

NBN VE vs pgmpy VE: 450 queries, max |diff| = 1.94e-07

nbn_cert: 528 queries with an exact reference; median |err| = 0.0019, max = 0.0069

nbn_cert: certified interval covers exact answer in 528/528 queries; median max-width 0.033

nbn_lw: 568 queries with an exact reference; median |err| = 0.0027, max = 0.0109

nbn_lw@master: 137 queries with an exact reference; median |err| = 0.0027, max = 0.0082

nbn_cert with bound='kl' only (earlier run): coverage 518/518; median max-width 0.067

nbn_lw (continuous, vs 4M-particle reference): 160 queries; median |err| = 0.0051, 90% = 0.0193

nbn_lw@master (continuous, vs 4M-particle reference): 150 queries; median |err| = 0.0060, 90% = 0.0211

pyro_vec (continuous, vs 4M-particle reference): 150 queries; median |err| = 0.0044, 90% = 0.0217
