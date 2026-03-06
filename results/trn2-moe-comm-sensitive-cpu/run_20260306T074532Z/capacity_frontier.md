# MoE Capacity Frontier

| Setup | Context | Experts | Top-k | Routing skew | SLO (ms) | Max tested conc | Max feasible conc | Best throughput (tokens/s) | Best conc | Best p90 latency (ms) | Has feasible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dual_die_moe_locality | 2048 | 8 | 2 | 0.00 | 120.00 | 32 | 16 | 2734.86 | 16 | 94.8639 | True |
| dual_die_moe_locality | 2048 | 8 | 2 | 2.50 | 120.00 | 32 | 16 | 2689.11 | 16 | 95.4469 | True |
| dual_die_moe_locality | 4096 | 8 | 2 | 0.00 | 120.00 | 32 | 16 | 2675.19 | 16 | 95.9383 | True |
| dual_die_moe_locality | 4096 | 8 | 2 | 2.50 | 120.00 | 32 | 16 | 2695.11 | 16 | 95.8043 | True |
| dual_die_moe_naive | 2048 | 8 | 2 | 0.00 | 120.00 | 32 | 16 | 2701.05 | 16 | 95.6610 | True |
| dual_die_moe_naive | 2048 | 8 | 2 | 2.50 | 120.00 | 32 | 16 | 2754.55 | 16 | 93.5780 | True |
| dual_die_moe_naive | 4096 | 8 | 2 | 0.00 | 120.00 | 32 | 16 | 2691.41 | 16 | 96.0253 | True |
| dual_die_moe_naive | 4096 | 8 | 2 | 2.50 | 120.00 | 32 | 16 | 2696.03 | 16 | 95.6049 | True |
| single_die | 2048 | 8 | 2 | 0.00 | 120.00 | 32 | 0 | 0.00 | 0 | 0.0000 | False |
| single_die | 2048 | 8 | 2 | 2.50 | 120.00 | 32 | 0 | 0.00 | 0 | 0.0000 | False |
| single_die | 4096 | 8 | 2 | 0.00 | 120.00 | 32 | 0 | 0.00 | 0 | 0.0000 | False |
| single_die | 4096 | 8 | 2 | 2.50 | 120.00 | 32 | 0 | 0.00 | 0 | 0.0000 | False |
