# Mixed Traffic Summary

- duration_s: 90.00
- arrival_rate_rps: 10.000
- prefill_ratio: 0.300
- decode_slo_ms: 250.00
- drop_wait_ms: 2000.00

| Policy | Goodput (tokens/s) | Served (tokens/s) | On-time % | Drop % | p50 latency (ms) | p90 latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single->single | 8172.09 | 11854.22 | 57.56 | 0.00 | 207.39 | 526.21 |
| single->request | 11303.82 | 11854.22 | 91.38 | 0.00 | 94.61 | 235.59 |
| single->tensor | 159.29 | 4534.04 | 1.16 | 61.37 | 2180.08 | 2346.36 |
| request->request | 11276.80 | 11854.22 | 90.93 | 0.00 | 98.70 | 241.71 |
