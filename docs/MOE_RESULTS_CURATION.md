# MoE Results Curation (For Paper Figures)

This note records which MoE plots are high signal and worth keeping in the public repo.

## Mainline Run

- Trainium-stable compact sweep: `results/trn2-moe-stable-small-merged-mask23/run_20260306T100500Z`

## Appendix-Only Runs

- CPU compute-heavy matrix: `results/trn2-moe-service-day1-cpu/run_20260306T065352Z`
- CPU comm-sensitive matrix: `results/trn2-moe-comm-sensitive-cpu/run_20260306T074532Z`

## Plot Review

### Keep (high signal, public-facing)

- `results/plots/public_moe_mask23_remote_dispatch_ratio.png`
  - Best direct evidence that locality routing reduces cross-rank dispatch on Trn2.
- `results/plots/public_moe_mask23_locality_gain.png`
  - Shows locality improves the dual path versus naive, even when single-die remains faster overall.
- `results/plots/public_moe_mask23_decode_slo_frontier.png`
  - Useful as a compact shape-by-shape reference for the committed Trn2 run.
- `results/plots/public_moe_mask23_capacity_frontier.png`
  - Useful companion for concurrency feasibility in the compact run.

### Keep only in appendix / exploratory notes

- `trn2_moe_day1_cpu_*`
  - Counterexample only. Every SLO row is infeasible, so this should not be a headline artifact.
- `trn2_moe_comm_sensitive_cpu_*`
  - Useful for exploration, but CPU-only and should not carry the main Trn2 narrative.

## Important runtime note

On this `trn2.3xlarge` host session, repeated 2-rank Neuron/XLA runs intermittently failed with Neuron runtime init/collective errors (`NRT_FAILURE`, `NRT_COLL_PENDING`) unrelated to algorithm correctness. The committed Trn2 MoE artifact was produced by running stable masked-core sweeps and merging the successful result slices.
