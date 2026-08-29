# AtypEmu all-label one-shared-q autoresearch

## Objective

Maximize `all_label_macro_one_shared_q_ccc` and stop immediately when a checks-passing candidate reaches `>= 0.95`. This is unseen-entry NMR evidence assimilation, not blind next-label prediction: each evaluation entry's assigned `D_e` is intentionally available to its NMR-conditioned coordinate generator and its single entity-level posterior `q_e`, and the posterior means are scored against the same `D_e`.

The development benchmark is bidirectional observer-fold crossfit (`A -> B` and `B -> A`) over the frozen A/B cohort. Repeated autoresearch makes this a model-selection surface, not a final sealed claim. When development reaches `0.95`, freeze the candidate and start no new hypothesis; a separate never-before-opened entry cohort is required for the final scientific claim.

## Primary metric

The sole ranking metric is the unweighted mean of held-out CCC values across the 62 prediction-independent eligible canonical BMRB `Atom_ID` labels frozen in `.auto/frozen/all_label_inventory.json`. The evaluator independently computes

`prediction[e,t] = sum_k q[e,k] * support_prediction[e,t,k]`

from one `q_e` table per entry. It does not accept candidate-supplied posterior means. `Comp_ID x Atom_ID`, broad-family, pooled, masked-label, min-family, and min-label metrics are diagnostic only and must never rank candidates or stop the loop.

- Every one of the 127,285 finite assigned rows must be present on every support member.
- Constant predictions remain scored. Missing or nonfinite support/predictive values invalidate the candidate; they never remove a label.
- Sparse `HH11` (one row) is predeclared ineligible and reported only as coverage.
- A different `q` by atom, family, residue, target row, or label is forbidden.

## Scientific validity backpressure

A high metric is not sufficient. `.auto/checks.sh` must pass before `keep`:

1. A/B train and evaluation entity sets exactly follow the frozen observer folds and never include the canonical outer sealed entries.
2. Assigned `D_e` is consumed by the coordinate generator and by one entity-level `q_e`; this is intended and must not be replaced with a target-unread contract.
3. The observer/surface reads generated complete coordinates, sequence/context, and frozen model state, but never reads assigned target values directly. With coordinates frozen, perturbing target values must leave the observer surface byte/numerically invariant.
4. Chemical-shift loss has nonzero gradients to parameters that alter complete coordinates, and conditioned versus no-evidence coordinate audits show finite nonzero movement. A torsion-feature edit with no emitted coordinate change is invalid.
5. Readout-only fitting, fixed-support reweighting alone, target copying into the surface, invalid coordinates, pooled-scale inflation, and oracle support/label selection are invalid regardless of CCC.
6. Candidate surfaces and q tables are byte-hashed; the frozen evaluator recomputes the metric and the separate arithmetic verifier must agree.

## Commands and outputs

Run `.auto/measure.sh`. It calls `gpuopt/run_all_label_candidate.py` and expects:

- `.auto/runs/current/surface_A_to_B.parquet`
- `.auto/runs/current/surface_B_to_A.parquet`
- `.auto/runs/current/q_A_to_B.parquet`
- `.auto/runs/current/q_B_to_A.parquet`
- matched `surface_no_coordinate_*.parquet` and `q_uniform_*.parquet` controls
- `.auto/runs/current/validity.json`
- `.auto/runs/current/coordinate_audit_A_to_B.npz`
- `.auto/runs/current/coordinate_audit_B_to_A.npz`

The measure script prints exactly one primary `METRIC all_label_macro_one_shared_q_ccc=<number>` line after the frozen evaluator and independent verifier pass.

## Files in scope

Editable candidate surface:

- `gpuopt/run_all_label_candidate.py`
- `gpuopt/candidates/**`

Off-limits for all experiments:

- `goal.md`
- `.auto/prompt.md`, `.auto/measure.sh`, `.auto/checks.sh`, `.auto/config.json`
- `.auto/frozen/**`
- `gpuopt/all_label_one_shared_q_metric.py`
- `gpuopt/test_all_label_one_shared_q_metric.py`
- `gpuopt/verify_all_label_one_shared_q_score.py`
- `gpuopt/check_all_label_candidate_validity.py`
- Parent `src/`, `scripts/`, `tests/`, `data/`, and historical outputs (read-only inputs)

Do not edit metric code, inventory, eligibility, cohort, checkpoints, or primary formula after observing results. Do not launch or inspect the outer sealed cohort during autoresearch.

## Loop discipline

1. State one hypothesis tied to a mechanism that can alter complete coordinates or shared posterior inference.
2. Make the smallest candidate-only change.
3. Run `.auto/measure.sh` through autoresearch tooling; run checks before `keep`.
4. Rank only by the primary metric. Record diagnostic evidence without using it to choose winners.
5. Keep a candidate only if checks pass and the primary metric improves; otherwise discard and retain the learning in the autoresearch log.
6. At `>= 0.95`, freeze the candidate and stop this development loop. At 100 experiments without success, stop and report the blocker rather than silently changing the contract.
