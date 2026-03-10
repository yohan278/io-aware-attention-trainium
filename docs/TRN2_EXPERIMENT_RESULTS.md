# TRN2 Experiment Results

This note summarizes the current Trainium-class evidence used by the paper package.

## Headline Findings

1. `dual_die_request_sharded` improves decode throughput and latency over `single_die` in the committed decode artifact.
2. `dual_die_tensor_optimized` is consistently worse than `single_die` in the same decode regime.
3. Prefill still prefers `single_die`.
4. The deployment lesson is phase-aware and request-aware: request-sharded decode is the key ingredient; exact single-prefill vs request-prefill ordering depends on workload mix.

## Direct End-to-End Policy Trace (Measured)

Direct trace run (`context_len=2048`, `batch/concurrency=16`, `output_tokens=128`, `request_slo_ms=1500`), comparing:

- `single->single`
- `single->request`
- `request->request`

| Policy | p50 latency (ms) | p90 latency (ms) | Prefill p50 (ms) | Decode p50 (ms) | Requests/s | On-time % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `single->single` | 1830.15 | 1841.06 | 53.11 | 1776.80 | 8.74 | 0.00 |
| `single->request` | 1388.49 | 1399.86 | 53.20 | 1335.30 | 11.52 | 100.00 |
| `request->request` | 1376.47 | 1389.70 | 28.38 | 1348.10 | 11.62 | 100.00 |

Relative to `single->single`, both request-aware policies are clearly better:

- p50 request throughput: `1.32x` (`single->request`) and `1.33x` (`request->request`)
- p50 latency ratio: `0.76x` (`single->request`) and `0.75x` (`request->request`)
- on-time service: `0%` -> `100%` under the same `1500 ms` SLO

## Decode Summary (Measured)

At context length `2048`:

| Setup | Concurrency | p50 latency (ms) | Throughput (tok/s) |
| --- | ---: | ---: | ---: |
| `single_die` | 8 | 48.70 | 657.02 |
| `dual_die_request_sharded` | 8 | 43.42 | 736.93 |
| `dual_die_tensor_optimized` | 8 | 173.64 | 184.29 |
| `single_die` | 16 | 52.78 | 1212.54 |
| `dual_die_request_sharded` | 16 | 40.72 | 1571.61 |
| `dual_die_tensor_optimized` | 16 | 184.21 | 347.44 |

Derived ratios versus `single_die`:

- request-sharded decode throughput: `1.12x` at `C=8`, `1.30x` at `C=16`
- request-sharded decode latency: `0.89x` at `C=8`, `0.77x` at `C=16`
- tensor-split decode throughput: `0.28x–0.29x`
- tensor-split decode latency: `3.49x–3.57x`

## Prefill Summary (Measured)

At sequence length `4096`, batch `2`:

- `single_die`: `36.72 ms`
- `dual_die_request_sharded`: `40.61 ms` (`1.11x` slower)
- `dual_die_tensor_optimized`: `1646.05 ms` (`44.8x` slower)

## Mixed-Traffic Summary (Simulated)

The mixed-traffic simulator uses measured phase profiles. Under the committed workload mix:

| Policy | Goodput (tok/s) | On-time % | p90 latency (ms) |
| --- | ---: | ---: | ---: |
| `single->single` | 8172.09 | 57.56 | 526.21 |
| `single->request` | 11303.82 | 91.38 | 235.59 |
| `request->request` | 11276.80 | 90.93 | 241.71 |
| `single->tensor` | 159.29 | 1.16 | 2346.36 |

`single->request` improves goodput by `1.38x` over `single->single`.

## Mechanism Summary

The kernel and collective evidence point to collective structure rather than byte volume as the main tensor-split bottleneck:

- representative optimized attention microbenchmark: `105.7 ms`, about `52%` communication time
- representative naive dual attention microbenchmark: `4.3 ms`, about `76%` communication time
- supporting phase artifact at prefill `seq=4096`:
  - `all_gather`: ~`128 MB`, ~`9.7 ms`
  - `all_reduce_max`: ~`4 MB`, ~`1245 ms`

## Evidence Types

- **Measured:** decode frontier, prefill ratio, attention comm breakdown, direct policy trace
- **Composed:** hybrid policy figure
- **Simulated:** mixed-traffic goodput

The paper and captions should keep these categories explicit.
