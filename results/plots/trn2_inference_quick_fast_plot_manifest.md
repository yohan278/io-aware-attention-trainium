# Inference Plot Manifest

This set keeps only plots that directly support inference chip advice.

- `trn2_inference_quick_fast_prefill_ratio.png`: Shows whether dual setup improves long-context prefill latency relative to single.
- `trn2_inference_quick_fast_decode_slo_frontier.png`: Shows throughput-at-SLO, the right lens for serving value.
- `trn2_inference_quick_fast_decode_kv_efficiency.png`: Shows tokens/s achieved per GiB KV footprint to capture capacity efficiency.
- `trn2_inference_quick_fast_comm_breakdown.png`: Shows whether dual performance is compute-limited or communication-limited.
