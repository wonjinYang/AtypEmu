# AtypEmu

AtypEmu is a BioEmu-centered research codebase for NMR-conditioned conformer
generation of intrinsically disordered protein ensembles.

The current `goal.md` route is not a direct chemical-shift predictor and not a
rowwise reweighting baseline.  The default path trains and audits an
end-to-end AtypEmu/BioEmu generated-support model:

```text
online GPU trainer
  -> immutable READY/checkpoint manifest
  -> asynchronous generated conformer support
  -> offline UCBShift sidecars
  -> one shared-q posterior inference
  -> posterior mean chemical shifts
  -> all-family CCC / validity / diversity / prior-drift gates
```

The final acceptance metric is offline UCBShift-passed, one shared-q
co-satisfaction CCC, with target CCC `>= 0.95` for every configured atom family.
Fast chemical-shift heads and student models are conservative training proxies;
they are never the final acceptance source.

This layout and documentation style intentionally follow the spirit of
well-structured research repositories such as DeepMind's AlphaFold: start with
the problem statement, make the repository layout explicit, keep setup and
execution paths reproducible, and document assumptions close to the code.

## Repository Status

Current status:

- The active contract is documented in [goal.md](goal.md).
- Online GPU training is separated from offline UCBShift evaluation.
- Online jobs target `l40sq` / `iREMB-C-08` with one GPU and do not wait on
  UCBShift.
- Offline teacher generation targets `gpuq` for CPU-heavy UCBShift sharding,
  merge, shared-q simplex, promotion, and viewer refresh.
- Promotion requires generated-support lineage, raw UCBShift-facing geometry,
  offline UCBShift posterior means, one shared support-level `q`, and strict
  posterior health gates.
- Legacy candidate-free, rowwise, direct CS-reweighting, isolated HN-tail,
  inference-overlay, rare-regime, and pre-v1219 x0 experiment routes are kept
  out of the default path through archive manifests.
- The current non-destructive cleanup inventory is
  [archive/goal_20260610_legacy_inventory](archive/goal_20260610_legacy_inventory/README.md).

## Cleanup Boundary

Repository cleanup is intentionally non-destructive because the active worktree
contains many experiment files and bridge diagnostics.  Use the archive helper
to refresh the current `goal.md` inventory:

```bash
python scripts/archive_atypemu_goal_legacy_assets.py \
  --include-unclassified \
  --stage-archive \
  --fail-on-active-legacy-references \
  --fail-on-stage-errors
```

This copies legacy candidates into `archive/goal_20260610_legacy_inventory/`
and writes manifests, but it does not delete or move source files.  Physical
`git mv`, `mv`, or `git worktree remove` cleanup is allowed only after the
generated manifests show no active-spine blockers and the target worktree is not
the current dirty checkout.

## Core ideas

AtypEmu treats a finite conformer batch through the family of weighted
empirical measures it supports and assigns it a profiled multi-observable NMR
energy rather than treating ensemble weights as the
final prediction target. In practical terms, the repository currently provides
the tooling needed to:

1. generate conformer support conditioned on chemical-shift evidence,
2. preserve BioEmu prior semantics and UCBShift-evaluable raw geometry,
3. compute offline UCBShift sidecars for generated conformers,
4. infer one shared support-level posterior `q`,
5. evaluate posterior mean chemical shifts by atom family,
6. and feed accepted teacher bundles back to the online trainer only at safe
   reload boundaries.

The underlying project plan is documented in [plan.md](plan.md), the formal
theory note is in [theory/main.tex](theory/main.tex), and a Korean
tutorial-style walkthrough is in [theory/ko/README.md](theory/ko/README.md).

## Repository layout

The repository uses a `src/` layout and keeps data, code, batch scripts, and
theory documents separate.

```text
AtypEmu/
├── README.md
├── plan.md
├── pyproject.toml
├── src/
│   └── atypemu/
│       ├── adapters/
│       ├── cli/
│       ├── databases/
│       ├── datasets/
│       ├── energy/
│       ├── evaluation/
│       ├── integrated/
│       ├── observables/
│       ├── reweighting/
│       ├── structures/
│       ├── errors.py
│       └── types.py
├── tests/
├── data/
│   ├── README.md
│   ├── bmrb/
│   ├── integrated/
│   ├── meta/
│   └── ped/
├── docker/
├── sbatch/
├── scripts/
└── theory/
    ├── main.tex
    └── ko/
```

## Python package structure

The Python package is intentionally split by responsibility.

- `atypemu.types`
  - public dataclasses and serialization helpers
- `atypemu.databases`
  - source-specific public database ingestion and normalization packages
- `atypemu.structures`
  - staged pool construction and structure-level metadata helpers
- `atypemu.observables`
  - chemical-shift, JC, and NOE observable builders
- `atypemu.reweighting`
  - MaxEnt and Euclidean simplex solvers
- `atypemu.energy`
  - posterior-energy aggregation
- `atypemu.evaluation`
  - metrics and evaluation reports
- `atypemu.integrated`
  - cross-source workspace assembly, meta dataframe export, and training-bundle
    preparation
- `atypemu.datasets`
  - generic parquet manifest types and lazy dataset registries
- `atypemu.cli`
  - thin command-line wrappers around library functions
- `atypemu.adapters`
  - interfaces to external ecosystems such as `cnnls` and future BioEmu code

Design rules used throughout the package:

- one module, one primary responsibility,
- public record-like objects use dataclasses,
- public functions use explicit type hints,
- public functions, methods, and classes use Google-style docstrings,
- command-line modules stay thin and delegate work to library code,
- reusable logic lives in `src/atypemu`, not in ad hoc scripts.

## Installation

### Local editable install

From the repository root:

```bash
cd AtypEmu
python -m pip install -e .
```

### Development checks

```bash
cd AtypEmu
python -m pytest
python -m black --check src tests
```

### Containerized setup

Container definitions are stored in `docker/`.

- `docker/Dockerfile`
  - runtime image used to build the AtypEmu container
- `docker/atypemu.def`
  - Singularity definition used on the remote cluster
- `docker/requirements.txt`
  - Python dependencies installed into the container

## Data layout

The local data root is documented in [data/README.md](data/README.md).

At a high level:

- `data/AF`
  - AF3-derived conformers
- `data/CALVADOS`
  - CALVADOS2-derived conformers
- `data/BioEmu`
  - BioEmu-derived conformers
- `data/bmrb`
  - BMRB downloads, curated tables, bundle assets, and source-owned datasets
- `data/ped`
  - PED tables, model assets, datasets, and debug crawls
- `data/sasbdb`
  - SASBDB SAXS tables, profile assets, datasets, and debug crawls
- `data/mfib`
  - MFIB bound-complex tables, CIF assets, optional interface geometry, and
    datasets
- `data/meta`
  - cross-source bootstrap metadata in `tables/` and meta dataframe outputs in
    `datasets/`
- `data/integrated`
  - multi-source inventories, splits, trainer-facing datasets, run artifacts,
    and cross-source config files

Database roots follow one contract:

- root-level allowed items are `README.md`, `downloads/`, `tables/`,
  `datasets/`, `assets/`, and `_debug/`
- public biological outputs live only in `tables/`, `datasets/`, and `assets/`
- provenance-only artifacts live only in `_debug/`
- source-specific table names stay meaningful, while dataset bundle naming stays
  consistent across sources

## Typical workflows

### 1. Prepare BMRB targets

```bash
cd AtypEmu
PYTHONPATH=src python -m atypemu.cli.databases.bmrb.prepare \
  --bmrb-id-file data/integrated/configs/bmrb_ids.txt \
  --output-root data/bmrb
```

### 2. Set up the integrated workspace

```bash
cd AtypEmu
PYTHONPATH=src python -m atypemu.cli.integrated.setup_workspace \
  --data-root data \
  --bmrb-root data/bmrb \
  --output-root data/integrated
```

### 3. Export integrated training bundles

```bash
cd AtypEmu
PYTHONPATH=src python -m atypemu.cli.integrated.export_training_dataframes \
  --data-root data \
  --integrated-root data/integrated
```

This writes the staged training plan plus trainer-facing tables such as
`teacher_examples`, `observable_supervision`, `benchmark_entries`, and
`training_entry_view`.

### 4. Materialize offline teachers and train the first student

```bash
cd AtypEmu
PYTHONPATH=src python -m atypemu.cli.training.materialize_teachers \
  --data-root data \
  --integrated-root data/integrated \
  --config-json data/integrated/configs/teacher_materialization.json

PYTHONPATH=src python -m atypemu.cli.training.audit_readiness \
  --data-root data \
  --integrated-root data/integrated \
  --config-json data/integrated/configs/training_readiness_gate.json \
  --require-ready

PYTHONPATH=src python -m atypemu.cli.training.train_student \
  --data-root data \
  --integrated-root data/integrated \
  --config-json data/integrated/configs/student_training_config.json \
  --output-dir data/integrated/runs/student_local_smoke
```

The default teacher materialization config expects precomputed candidate
chemical-shift sidecars under `data/AF/shifts/<bmrb_id>`,
`data/BioEmu/shifts/<bmrb_id>`, and `data/CALVADOS/shifts/<bmrb_id>`. If those
sidecars are absent, teacher materialization will skip those accessions instead
of inventing supervision.

For UCBShift2.0-based sidecars and baseline tables, use:

```bash
PYTHONPATH=src python -m atypemu.cli.training.generate_ucbshift_sidecars \
  --data-root data \
  --integrated-root data/integrated \
  --config-json data/integrated/configs/ucbshift_sidecars.json
```

The remote helper `scripts/submit_iremb6_atypemu_ucbshift_sidecars.sh` prepares
the UCBShift2.0 checkout, downloads `models.zip` from Zenodo, generates
candidate sidecars, and writes
`data/integrated/baselines/ucbshift2_holdout.csv` plus
`data/integrated/baselines/ucbshift2_reference_corpus.csv` for report-only
baseline comparison.

Student training checkpoints are selected by a teacher-first metric policy:
`val_teacher_kl_macro`, then `val_teacher_js_macro`,
`val_cs_rmse_z_macro`, and `val_noe_violation_rate_macro`. Training runs also
write `benchmark_report.json`, `chemical_shift_baseline_report.json`, and
`uncertainty_report.json`. Static benchmark figures, representative annotated
structures, and posterior arrays are mirrored under `reports/figures`,
`reports/structures`, and `reports/arrays`; the baseline comparison stays
report-only and does not affect checkpoint selection.

For `iremb6-server`, the repository also ships campaign helpers:

- `scripts/submit_iremb6_atypemu_baseline_campaign.sh`
- `scripts/submit_iremb6_atypemu_cs_sweep.sh`

These scripts keep the production policy fixed to `iremb-c-08`, `l40sq`,
`gpu:1`, `cpus-per-task=16`, and sequential submissions that never touch other
users' jobs.

### 5. Crawl external benchmark sources

Bootstrap and crawl through reproducible bash wrappers:

```bash
cd AtypEmu
PYTHONPATH=src python -m atypemu.cli.databases.setup_source_workspace --data-root data
./scripts/external/crawl_all_external.sh
./scripts/external/crawl_external_source.sh PED
```

Detailed source-specific collection notes are in
[data/ped/README.md](data/ped/README.md),
[data/sasbdb/README.md](data/sasbdb/README.md),
[data/mfib/README.md](data/mfib/README.md),
and the source-level layout contract is summarized in
[data/README.md](data/README.md).

## Data contract audit

When adding or revising a database source:

- keep the root under `data/<source>/`
- use only `downloads/`, `tables/`, `datasets/`, `assets/`, and `_debug/`
- avoid nested `README.md` files below the source root
- keep public table names source-semantic
- keep dataset bundle names generic enough for the shared registries

### 6. Run the offline core

The current offline workflow is:

1. stage accession-level candidates,
2. build a candidate pool,
3. compute observable matrices,
4. fit simplex weights,
5. evaluate the resulting posterior energy.

Representative commands:

```bash
cd AtypEmu
PYTHONPATH=src python -m atypemu.cli.structures.stage_bmrb --help
PYTHONPATH=src python -m atypemu.cli.structures.build_pool --help
PYTHONPATH=src python -m atypemu.cli.offline.compute_observables --help
PYTHONPATH=src python -m atypemu.cli.offline.fit_weights --help
PYTHONPATH=src python -m atypemu.cli.offline.evaluate --help
```

## Remote deployment and Slurm execution

Remote execution support is organized into three layers.

- `scripts/`
  - developer-facing convenience wrappers
- `docker/`
  - build context for the AtypEmu runtime image
- `sbatch/`
  - cluster-facing Slurm scripts

Typical remote flow:

1. sync the repository snapshot,
2. optionally sync the core training data subset,
3. build or update the SIF,
4. run smoke tests,
5. materialize offline teachers,
6. launch the first density-student training job,
7. move on to BioEmu-specific preflight or fine-tuning.

Relevant helpers:

- `scripts/sync_iremb6_atypemu.sh`
- `scripts/sync_iremb6_atypemu_data.sh`
- `scripts/submit_iremb6_atypemu_setup.sh`
- `scripts/submit_iremb6_atypemu_offline.sh`
- `scripts/submit_iremb6_atypemu_training.sh`
- `scripts/submit_iremb6_atypemu_ucbshift_sidecars.sh`
- `scripts/watch_iremb6_atypemu.sh`
- `sbatch/iremb6_l40sq_atypemu_build_sif.sh`
- `sbatch/iremb6_l40sq_atypemu_smoke.sh`
- `sbatch/iremb6_l40sq_atypemu_offline_eval.sh`
- `sbatch/iremb6_l40sq_atypemu_train_student.sh`
- `sbatch/iremb6_l40sq_atypemu_ucbshift_sidecars.sh`
- `sbatch/iremb6_l40sq_atypemu_bioemu_train_preflight.sh`
- `sbatch/iremb6_l40sq_atypemu_bioemu_finetune.sh`

For the shared `iremb6-server` setup, the defaults now target
`~/scratch/yang07/atypemu`, submit through `sbatch`, pin to `iremb-c-08`, and
request one L40S-backed GPU with at most 16 CPUs per job.

## Development style guide

AtypEmu follows a deliberately conservative Python style.

### Code style

- PEP 8 naming, spacing, and module organization
- maximum line length: `88`
- `black` for formatting
- `ruff` configuration kept in `pyproject.toml`
- explicit type hints on public functions and methods
- Google-style docstrings for public modules, classes, and functions

### Comment style

- comments explain assumptions, invariants, or non-obvious math
- comments do not restate the code line by line
- CLI and shell scripts include concise header comments for purpose and usage

### File structure

- keep reusable logic inside `src/atypemu`
- keep structure-source preparation in `structures/`
- keep cross-source workspace assembly in `integrated/`
- keep command-line wrappers in `cli/`
- keep evaluation logic separate from training and parsing
- prefer small, responsibility-focused modules over large mixed-purpose files

### Documentation style

- README files explain purpose, layout, assumptions, and execution paths
- data subtrees document on-disk contracts close to the data they describe
- theory and implementation plans remain versioned alongside code

## Output and artifact structure

The repository keeps generated artifacts in predictable locations.

- `data/bmrb`
  - source-owned BMRB downloads, curated tables, parsed bundles, and parquet datasets
- `data/meta`
  - cross-source source catalogs, registries, and parquet metadata
- `data/integrated`
  - inventories, splits, staging trees, pools, observables, and run products
- `data/ped`
  - PED biological tables, model assets, and source-owned parquet datasets
- `theory`
  - formal mathematical specification and compiled PDF

## Known limitations

- BioEmu fine-tuning is not yet the default execution path in this repository.
- JC support is intentionally narrow in v1 and starts with `3J_HNHA`.
- NOE handling is conservative and limited to simple unique restraints.
- External-source crawling currently prioritizes curated metadata bundles and
  page snapshots over full bulk-ingestion pipelines.

## References

- PED: https://proteinensemble.org/
- MobiDB: https://mobidb.org/
- SASBDB: https://www.sasbdb.org/
- DIBS: https://dibs.pbrg.hu/downloads.php
- MFIB: https://mfib.pbrg.hu/downloads.php
- IDEAL: https://www.ideal-db.org/current.html
- FuzDB: https://fuzdb.org/
