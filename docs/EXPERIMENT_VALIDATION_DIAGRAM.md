# Experiment Validation Diagram

This document describes the exact execution and validation flow used for Trn2 dual-die experiments.

## End-to-End Flow

```mermaid
flowchart TD
    A["Start: Trn2 host + env validation"] --> B["Fabric calibration<br/>(ping-pong, all-reduce, all-gather)"]
    B --> C["Kernel study (5 kernels)<br/>single_die vs dual_die_naive vs dual_die_optimized"]
    C --> D{"Correctness gates pass?<br/>max_abs_err + robust max_rel_err"}
    D -- "No" --> E["Fail run and record diagnostics"]
    D -- "Yes" --> F["Phase study (inference path)<br/>prefill + decode"]
    F --> G{"Correctness gates pass?<br/>all setups and shapes"}
    G -- "No" --> E
    G -- "Yes" --> H["Emit run artifacts<br/>metrics.csv/jsonl, collectives_summary.json,<br/>fabric_calibration.json, manifest"]
    H --> I["Generate plots<br/>kernel, phase, inference-track"]
    I --> J["Break-even modeling<br/>T_dual = T_compute_split + T_comm * (1-overlap)"]
    J --> K["AWS chip advice output<br/>when dual-die wins, required link BW/latency/overlap"]
```

## Setup and Measurement Matrix

```mermaid
flowchart LR
    S1["single_die<br/>reference baseline"] --> M["Compare p50/p90 latency,<br/>throughput, comm/compute split"]
    S2["dual_die_naive<br/>high-collective reference"] --> M
    S3["dual_die_optimized<br/>reduced-collective design"] --> M
    S4["dual_die_request_sharded<br/>(phase study decode/prefill)"] --> M
```

## Validation Gates

- Gate 1: Numerical correctness against single-die reference.
- Gate 2: Run must include fabric calibration and collective summaries.
- Gate 3: Per-run manifest records setup, device, and configuration.
- Gate 4: Plot generation from run artifacts must complete without missing inputs.

## Why this is credible

- Separates wall time and communication time explicitly, with overlap reported only when it is directly observable or clearly marked as an estimate.
- Tracks communication bytes and calibration-relative effective link rate per setup.
- Uses both kernel-level and end-to-end phase-level measurements.
- Includes break-even modeling to map today’s data to next-gen chip requirements.
