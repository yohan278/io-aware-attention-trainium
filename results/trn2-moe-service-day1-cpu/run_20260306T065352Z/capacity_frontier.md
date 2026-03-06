# MoE Capacity Frontier

| Setup | Context | Experts | Top-k | Routing skew | SLO (ms) | Max tested conc | Max feasible conc | Best throughput (tokens/s) | Best conc | Best p90 latency (ms) | Has feasible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dual_die_moe_locality | 2048 | 8 | 2 | 0.00 | 250.00 | 24 | 0 | 0.00 | 0 | 0.0000 | False |
| dual_die_moe_locality | 2048 | 8 | 2 | 1.80 | 250.00 | 24 | 0 | 0.00 | 0 | 0.0000 | False |
| dual_die_moe_locality | 4096 | 8 | 2 | 0.00 | 250.00 | 24 | 0 | 0.00 | 0 | 0.0000 | False |
| dual_die_moe_locality | 4096 | 8 | 2 | 1.80 | 250.00 | 24 | 0 | 0.00 | 0 | 0.0000 | False |
| dual_die_moe_locality | 8192 | 8 | 2 | 0.00 | 250.00 | 16 | 0 | 0.00 | 0 | 0.0000 | False |
| dual_die_moe_locality | 8192 | 8 | 2 | 1.80 | 250.00 | 16 | 0 | 0.00 | 0 | 0.0000 | False |
| dual_die_moe_naive | 2048 | 8 | 2 | 0.00 | 250.00 | 24 | 0 | 0.00 | 0 | 0.0000 | False |
| dual_die_moe_naive | 2048 | 8 | 2 | 1.80 | 250.00 | 24 | 0 | 0.00 | 0 | 0.0000 | False |
| dual_die_moe_naive | 4096 | 8 | 2 | 0.00 | 250.00 | 24 | 0 | 0.00 | 0 | 0.0000 | False |
| dual_die_moe_naive | 4096 | 8 | 2 | 1.80 | 250.00 | 24 | 0 | 0.00 | 0 | 0.0000 | False |
| dual_die_moe_naive | 8192 | 8 | 2 | 0.00 | 250.00 | 16 | 0 | 0.00 | 0 | 0.0000 | False |
| dual_die_moe_naive | 8192 | 8 | 2 | 1.80 | 250.00 | 16 | 0 | 0.00 | 0 | 0.0000 | False |
| single_die | 2048 | 8 | 2 | 0.00 | 250.00 | 24 | 0 | 0.00 | 0 | 0.0000 | False |
| single_die | 2048 | 8 | 2 | 1.80 | 250.00 | 24 | 0 | 0.00 | 0 | 0.0000 | False |
| single_die | 4096 | 8 | 2 | 0.00 | 250.00 | 24 | 0 | 0.00 | 0 | 0.0000 | False |
| single_die | 4096 | 8 | 2 | 1.80 | 250.00 | 24 | 0 | 0.00 | 0 | 0.0000 | False |
| single_die | 8192 | 8 | 2 | 0.00 | 250.00 | 16 | 0 | 0.00 | 0 | 0.0000 | False |
| single_die | 8192 | 8 | 2 | 1.80 | 250.00 | 16 | 0 | 0.00 | 0 | 0.0000 | False |
