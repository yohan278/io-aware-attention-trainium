# MoE Capacity Frontier

| Setup | Context | Experts | Top-k | Routing skew | SLO (ms) | Max tested conc | Max feasible conc | Best throughput (tokens/s) | Best conc | Best p90 latency (ms) | Has feasible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dual_die_moe_locality | 2048 | 8 | 2 | 0.00 | 250.00 | 16 | 16 | 2984.82 | 16 | 126.6235 | True |
| dual_die_moe_locality | 2048 | 8 | 2 | 1.80 | 250.00 | 16 | 16 | 2984.22 | 16 | 43.6238 | True |
| dual_die_moe_locality | 4096 | 8 | 2 | 0.00 | 250.00 | 16 | 16 | 3009.11 | 16 | 43.0050 | True |
| dual_die_moe_locality | 4096 | 8 | 2 | 1.80 | 250.00 | 16 | 16 | 2975.08 | 16 | 43.7747 | True |
| dual_die_moe_naive | 2048 | 8 | 2 | 0.00 | 250.00 | 16 | 8 | 1513.25 | 8 | 123.4820 | True |
| dual_die_moe_naive | 2048 | 8 | 2 | 1.80 | 250.00 | 16 | 16 | 2745.42 | 16 | 47.5669 | True |
| dual_die_moe_naive | 4096 | 8 | 2 | 0.00 | 250.00 | 16 | 16 | 2979.93 | 16 | 43.8344 | True |
| dual_die_moe_naive | 4096 | 8 | 2 | 1.80 | 250.00 | 16 | 16 | 2952.26 | 16 | 43.7977 | True |
| single_die | 2048 | 8 | 2 | 0.00 | 250.00 | 16 | 8 | 27003.89 | 8 | 16.4309 | True |
| single_die | 2048 | 8 | 2 | 1.80 | 250.00 | 16 | 16 | 43363.91 | 16 | 3.0149 | True |
| single_die | 4096 | 8 | 2 | 0.00 | 250.00 | 16 | 16 | 43279.12 | 16 | 3.0401 | True |
| single_die | 4096 | 8 | 2 | 1.80 | 250.00 | 16 | 16 | 42110.27 | 16 | 3.2158 | True |
