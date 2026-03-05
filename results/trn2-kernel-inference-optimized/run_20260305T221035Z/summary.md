# Kernel Study Summary

| Kernel | Setup | Avg p50 latency (ms) | Avg p50 compute (ms) | Avg p50 comm (ms) | Avg link util (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| attention | dual_die_naive | 3.8246 | 0.9911 | 2.7870 | 501.92 |
| attention | dual_die_optimized | 74.7322 | 34.9750 | 39.6364 | 5.60 |
| attention | single_die | 0.7709 | 0.7709 | 0.0000 | 0.00 |
| attention | single_die_native | 0.9317 | 0.9317 | 0.0000 | 0.00 |
| mlp | dual_die_naive | 2.1983 | 0.9949 | 1.2423 | 330.37 |
| mlp | dual_die_optimized | 1.3277 | 0.2991 | 1.0327 | 198.54 |
| mlp | single_die | 0.7846 | 0.7846 | 0.0000 | 0.00 |
| out_proj | dual_die_naive | 2.6686 | 0.7641 | 1.8893 | 130.33 |
| out_proj | dual_die_optimized | 0.9865 | 0.2106 | 0.7762 | 269.62 |
| out_proj | single_die | 0.2597 | 0.2597 | 0.0000 | 0.00 |
| qkv_proj | dual_die_naive | 2.6410 | 0.6689 | 1.9800 | 311.09 |
| qkv_proj | dual_die_optimized | 1.6645 | 0.6392 | 1.0236 | 314.72 |
| qkv_proj | single_die | 0.4890 | 0.4890 | 0.0000 | 0.00 |
| rmsnorm | dual_die_naive | 1.3809 | 0.6417 | 0.7393 | 138.75 |
| rmsnorm | dual_die_optimized | 2.7042 | 0.8165 | 1.8950 | 56.83 |
| rmsnorm | single_die | 0.2414 | 0.2414 | 0.0000 | 0.00 |
