# Direct Policy Trace Summary

- batch: 16
- context_len: 2048
- output_tokens: 128
- request_slo_ms: 1500.00

| Policy | p50 latency (ms) | p90 latency (ms) | Prefill p50 (ms) | Decode p50 (ms) | Requests/s | Tokens/s | On-time % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single->single | 1830.15 | 1841.06 | 53.11 | 1776.80 | 8.74 | 1119.03 | 0.00 |
| single->request | 1388.49 | 1399.86 | 53.20 | 1335.30 | 11.52 | 1474.98 | 100.00 |
| request->request | 1376.47 | 1389.70 | 28.38 | 1348.10 | 11.62 | 1487.86 | 100.00 |
