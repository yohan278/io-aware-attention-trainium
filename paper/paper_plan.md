# Tight Paper Plan

## Goal

Produce a complete MLSys-style paper package whose central claim is:

> Dual-die value on Trainium-class accelerators comes from phase-aware serving and request-sharded decode, not tensor-parallel splitting of a single request.

## What Is Already Generated

### Manuscript

- `paper/main.tex`
- `paper/mlsys2026.sty`
- `paper/refs.bib`

### Figure Assets

- Existing measured service figures in `results/plots/public_service_*`
- Existing measured MoE figures in `results/plots/public_moe_mask23_*`
- New paper-only figures:
  - `paper/figures/project_overview_pipeline.png`
  - `paper/figures/chiplet_comm_scaling.png`
  - `paper/figures/headline_metrics_summary.png`

### Evidence and Provenance

- `paper/claims.md`
- `paper/figures/README.md`
- `paper/README.md`
- `paper/data/headline_metrics.json`
- `paper/data/chiplet_proxy_example.csv`

## Paper Structure

1. Introduction and motivation
2. Problem statement
3. Methodology and implementation
4. Experimental setup
5. Evaluation
6. Discussion and limitations
7. Related work
8. Conclusion

## Figure Plan

### Main-text figures

1. `project_overview_pipeline.png`
2. `chiplet_comm_scaling.png`
3. `public_service_decode_slo_frontier.png`
4. `public_service_prefill_ratio.png`
5. `public_service_comm_breakdown.png`
6. `public_service_hybrid_e2e.png`
7. `public_service_mixed_trace_goodput.png`

### Optional / appendix figures

- `headline_metrics_summary.png`
- `public_moe_mask23_locality_gain.png`
- `public_service_collective_count_vs_latency.png`

## Tables To Keep

1. Chiplet proxy correctness and communication table
2. Decode summary table
3. Mixed-traffic policy table

## Still Needed Before Submission

### Must generate

- A directly measured end-to-end policy trace for `single->single`, `single->request`, and `request->request`
- A build on a machine with `pdflatex` or `latexmk`
- Final author list, affiliations, and acknowledgements
- Final bibliography expansion beyond the seed references
- A refreshed canonical phase artifact after the fixed decode-collective aggregation path

### Strongly recommended

- Direct end-to-end hybrid trace rather than only composed phase estimates
- Larger decode/context sweep for a real regime map
- Chiplet proxy sweep over `S` and `Dv` with stored raw outputs, not only analytical formulas
- One polished appendix figure for MoE or remove MoE from main text entirely

### Nice to have

- Official MLSys template validation against a real conference style file
- Camera-ready figure export to PDF/SVG in addition to PNG
- Teaser figure placement tuning after the first successful LaTeX build

## Risks

- The current compact Trainium artifact is strong enough for a project report, but borderline thin for a full systems paper without an expanded sweep.
- The hybrid figure is explicitly composed, so the paper must never describe it as directly measured.
- The chiplet proxy numbers are currently represented in paper data files and report notes, not a live benchmark script inside this repository snapshot.
