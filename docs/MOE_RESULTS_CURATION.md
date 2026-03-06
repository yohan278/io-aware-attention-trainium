# MoE Results Curation (For Paper Figures)

This note records which generated MoE plots are high signal and should be kept in-repo.

## Runs

- Compute-heavy matrix: `results/trn2-moe-service-day1-cpu/run_20260306T065352Z`
- Comm-sensitive matrix: `results/trn2-moe-comm-sensitive-cpu/run_20260306T074532Z`

## Plot Review

### Keep (high signal)

- `results/plots/trn2_moe_comm_sensitive_cpu_decode_slo_frontier.png`
  - Best headline figure: shows dual setups meeting SLO where single has fewer feasible points.
- `results/plots/trn2_moe_comm_sensitive_cpu_capacity_frontier.png`
  - Shows feasible concurrency expansion in comm-sensitive setup.
- `results/plots/trn2_moe_comm_sensitive_cpu_remote_dispatch_ratio.png`
  - Direct evidence that locality placement reduces cross-rank routing.
- `results/plots/trn2_moe_comm_sensitive_cpu_comm_breakdown.png`
  - Explains where time goes (compute vs communication) for the winning regime.
- `results/plots/trn2_moe_day1_cpu_comm_breakdown.png`
  - Counterexample: compute-heavy regime where communication optimization has limited impact.
- `results/plots/trn2_moe_day1_cpu_remote_dispatch_ratio.png`
  - Confirms locality reduces remote routing even when end-to-end speedup is modest.
- `results/plots/trn2_moe_comm_sensitive_cpu_summary.md`
- `results/plots/trn2_moe_day1_cpu_summary.md`

### Exclude from curated paper figure set (lower signal / redundant)

- `trn2_moe_day1_cpu_decode_slo_frontier.png`
  - SLO thresholds are too strict for this heavy config, producing low decision value.
- `trn2_moe_day1_cpu_capacity_frontier.png`
  - Redundant with decode SLO and comm breakdown in this run.
- `trn2_moe_day1_cpu_locality_gain.png`
  - Nearly flat around parity; less informative than remote-dispatch + comm breakdown.
- `trn2_moe_comm_sensitive_cpu_locality_gain.png`
  - Adds little beyond summary table + remote-dispatch figure.

## Important runtime note

On this specific `trn2.3xlarge` host session, repeated 2-rank Neuron/XLA runs intermittently failed with Neuron runtime init/collective errors (`NRT_FAILURE`, `NRT_COLL_PENDING`) unrelated to algorithm correctness. CPU-distributed runs completed and are included as reproducible evidence.
