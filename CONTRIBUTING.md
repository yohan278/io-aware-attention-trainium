# Contributing

## Development Setup

```bash
conda env create -f conda/environment.cpu.yml
conda activate ioattn-trn2
python -m pip install -e .
```

## Required Checks

Run before opening a PR:

```bash
ruff check .
pytest -q
```

## Contribution Guidelines

- Keep changes focused and scoped to a clear goal.
- Add/update tests for behavior changes.
- Do not commit generated outputs from `results/`.
- Keep configs reproducible and document new flags in `docs/EXPERIMENT_GUIDE.md`.
