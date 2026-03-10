# Repo Cleanup and Paper Plan

## Current Paper-Ready Assets

- MLSys-style draft: `paper/main.tex`
- local style file: `paper/mlsys2026.sty`
- figure provenance: `paper/figures/README.md`
- claim map: `paper/claims.md`
- tight execution plan: `paper/paper_plan.md`
- current evidence summary: `docs/TRN2_EXPERIMENT_RESULTS.md`

## Cleanup Priorities

1. Keep the current public figures as the canonical paper figure set.
2. Keep all paper-specific figures under `paper/figures/`.
3. Keep all derived paper data under `paper/data/`.
4. Keep the measured/composed/simulated distinction explicit in docs and captions.

## What Still Needs Generation

### Required for a cleaner submission package

- PDF build on a machine with LaTeX installed
- author metadata
- final references pass
- refreshed phase artifact after decode collective-attribution fix

### Recommended for stronger evidence

- larger decode/context sweep
- direct end-to-end hybrid measurement
- chiplet-proxy sensitivity sweep over larger `S` and `Dv`

## Narrative Discipline

The paper should say:

- request-sharded dual-die improves decode serving
- prefill stays on single-die
- tensor-split dual-die is communication-bound

The paper should not say:

- dual-die helps every inference path
- the hybrid estimate is directly measured
- MoE locality overturns the main result
