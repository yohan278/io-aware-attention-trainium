# Paper Draft

This directory contains an MLSys-style LaTeX draft built around the currently committed Trainium artifacts and the broader project narrative on IO-aware attention for non-GPU and multi-die accelerators.

The paper package is self-contained: the current paper figures are copied into `paper/figures/` so the manuscript does not need to reference `results/plots/` at build time.

## Build

```bash
cd paper
latexmk -pdf main.tex
```

If `latexmk` is unavailable, use:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Evidence Types

- **Measured:** `results/trn2-phase-inference-quick-fast/run_20260305T224828Z/*`
- **Measured direct trace:** `results/trn2_direct_policy_trace/run_20260310T044926Z/*`
- **Measured microbenchmark:** `results/trn2-kernel-inference-optimized/run_20260305T221035Z/*`
- **Composed:** `results/plots/public_service_hybrid_e2e.png`
- **Simulated:** `results/plots/public_service_mixed_trace_goodput.png`
- **Secondary MoE evidence:** `results/trn2-moe-stable-small-merged-mask23/run_20260306T100500Z/*`

## Support Files

- tight execution plan: `paper/paper_plan.md`
- claim-to-artifact map: `paper/claims.md`
- figure regeneration: `paper/figures/README.md`
- seed bibliography: `paper/refs.bib`
- local MLSys-style formatting: `paper/mlsys2026.sty`

See `paper/claims.md` for claim-to-artifact mapping and `paper/figures/README.md` for figure regeneration commands.
