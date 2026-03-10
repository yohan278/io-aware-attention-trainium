# Dense Sharded Serving Analysis

## Direct measured decode and end-to-end comparison

- End-to-end median latency: `1719.79 ms` (single) vs `1356.07 ms` (single->request), speedup `1.27x`, absolute drop `363.72 ms`.
- Decode median latency component: `1667.75 ms` (single) vs `1302.52 ms` (single->request), speedup `1.28x`, absolute drop `365.23 ms`.

## Multi-user queue simulation from measured service-time samples

| Policy | Max arrival rate with >=90% on-time | Goodput at that point (tok/s) |
| --- | ---: | ---: |
| single->single | 6.0 req/s | 737.5 |
| single->request | 9.0 req/s | 1087.3 |
