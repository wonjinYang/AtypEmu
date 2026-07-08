# AtypEmu

AtypEmu is a Python package for ensemble-based NMR observable modeling.  It
provides reusable library code and command-line entry points for preparing
protein ensemble data, computing observables, fitting posterior weights, and
evaluating ensemble predictions.

The public repository is intentionally package-first.  Local research notes,
experiment launchers, logs, generated data, and analysis artifacts are not part
of the deployment package.

## Install

From the repository root:

```bash
python -m pip install -e .
```

The package requires Python 3.10 or newer.

Optional extras:

```bash
python -m pip install -e '.[crawl]'
python -m pip install -e '.[monitor]'
python -m pip install -e '.[render]'
```

## Package Layout

```text
AtypEmu/
├── README.md
├── pyproject.toml
├── docker/
└── src/
    └── atypemu/
        ├── adapters/
        ├── cli/
        ├── databases/
        ├── datasets/
        ├── energy/
        ├── evaluation/
        ├── integrated/
        ├── monitoring/
        ├── observables/
        ├── reweighting/
        ├── structures/
        └── training/
```

## Main Modules

- `atypemu.databases`: source-specific ingestion and normalization helpers.
- `atypemu.structures`: structure staging, archive unpacking, and pool builders.
- `atypemu.observables`: chemical-shift, J-coupling, and NOE observable helpers.
- `atypemu.reweighting`: posterior weight solvers for ensemble fitting.
- `atypemu.energy`: posterior-energy aggregation.
- `atypemu.evaluation`: evaluation metrics and report helpers.
- `atypemu.integrated`: cross-source workspace assembly and dataframe export.
- `atypemu.training`: training utilities, chemical-shift predictors, and
  posterior-inference helpers.
- `atypemu.cli`: command-line wrappers around package functions.

## Command-Line Entry Points

Installed commands are defined in `pyproject.toml`.  Common entry points:

```bash
atypemu-build-targets
atypemu-stage-bmrb
atypemu-unpack-archives
atypemu-prepare-bmrb
atypemu-export-bmrb-dataframes
atypemu-setup-training-workspace
atypemu-setup-source-workspace
atypemu-crawl-source-assets
atypemu-materialize-teachers
atypemu-audit-training-readiness
atypemu-train-student
atypemu-generate-ucbshift-sidecars
atypemu-export-cs-reweighting-teacher
atypemu-merge-ucbshift-sidecars
atypemu-audit-cs-prediction-parity
atypemu-monitor-training
atypemu-build-pool
atypemu-compute-observables
atypemu-fit-weights
atypemu-evaluate
```

Example:

```bash
atypemu-evaluate --help
```

## Container Files

Container build files are kept under `docker/`:

- `docker/Dockerfile`
- `docker/atypemu.def`
- `docker/requirements.txt`

## Development Check

For a quick source-level check:

```bash
python -m compileall -q src
```

Project-specific tests and experiment assets may exist in local worktrees, but
they are outside the deployment tracking set.
