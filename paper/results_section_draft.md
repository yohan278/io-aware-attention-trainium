# Results Section Draft

## Main result

The main empirical result is phase asymmetry. Request-sharded dual-die helps decode service, while tensor-split dual-die hurts both latency and throughput. In the committed decode artifact at context 2048, request-sharded dual-die improves throughput by `1.12x` at concurrency 8 and `1.30x` at concurrency 16, while reducing p50 latency to `0.89x` and `0.77x` of single-die respectively. Tensor-split dual-die instead drops throughput to `0.28x–0.29x` of single-die and increases p50 latency by `3.49x–3.57x`.

## Why this matters

This changes the deployment story. The useful abstraction for dual-die is not tensor parallelism over one request. The useful abstraction is phase-aware serving: keep prefill on single-die, then move decode to request-sharded dual-die when concurrency is high enough.

## Supporting evidence

### Prefill

Prefill remains a single-die regime in the committed artifact. At sequence length 4096, request-sharded dual-die is still `1.11x` slower than single-die in p50 latency, while tensor-split dual-die is `44.8x` slower. This is why a static dual-die policy is not defensible.

### Mechanism

The representative attention microbenchmark shows that the optimized tensor-split path is dominated by collective time. In the supporting phase artifact, `all_reduce_max` totals only about 4 MB but consumes about 1245 ms, while `all_gather` totals about 128 MB in only about 9.7 ms. The important variable is therefore collective structure and latency class, not only bytes moved.

### Policy-level outcome

The direct end-to-end policy trace confirms the headline claim at serving scale. At `context=2048`, `concurrency=16`, `output_tokens=128`, both request-aware policies improve p50 request throughput by `1.32x–1.33x` over `single->single` and move on-time service from `0%` to `100%` under a `1500 ms` SLO. The composed hybrid policy and mixed-traffic simulator remain useful for broader sweeps; in the committed mixed trace, `single->request` improves goodput by about `1.38x` over `single->single`, with on-time service increasing from `57.6%` to `91.4%`.

## Secondary result

The stable MoE artifact supports a narrower claim. Locality-aware routing improves the dual-die MoE path over naive routing by a median of about `1.03x` and reduces remote dispatch by a median absolute fraction of `0.1875`. This is useful secondary evidence for the communication-structure story, but it does not replace the phase-aware decode result as the main contribution.
