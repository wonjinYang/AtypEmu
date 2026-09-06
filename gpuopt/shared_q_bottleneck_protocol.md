# Shared-q bottleneck protocol

Status: **DRAFT / HOLD — not an execution plan, source commitment, or authorization.**

This document turns the reviewed `comments.md`, `theory.md`, and
`shared_q_support_diagnostic.py` into an ordered research decision. It does not
change `goal.md`, open targets, or permit a science run.

## Goal

Before increasing support count, model width, or optimizer steps, distinguish:

1. incomplete all-atom inputs,
2. fixed-observer support capacity,
3. the intended KL tradeoff,
4. numerical posterior optimization,
5. missing coordinate-to-observer response, and
6. observer/calibration error.

No diagnostic below is a macro-CCC ceiling or an independent structure
validation.

## Prerequisite HOLD

Do not materialize a target-reading diagnostic until all of these are frozen and
independently checked:

- one complete-coordinate topology and observation meaning for every positive-mass
  support and every retained assigned label;
- explicit protonation/condition treatment for 119 resolved and 16
  missing/ambiguous source entities, without outcome-driven imputation;
- fixed source folds, label eligibility, normalization, row weights, observer,
  support IDs and prior mass; and
- a separately named source-only plan, source-byte commitment, cold review, and
  fresh once-only authorization.

The current protonation and support-count work remains target-unread and HOLD-only.
Synthetic self-tests in `shared_q_support_diagnostic.py` do not satisfy these
prerequisites.

## Source-only diagnostic arms

For each source-crossfit direction and entry, bind one matrix `A` whose rows are
the full frozen observation cohort and whose columns are complete conformers. All
arms use byte-identical `A`, `y`, row order, weights, normalization and support
prior. A missing geometry value fails the entry; it must not remove a support and
renormalize q for only one label.

1. **WLS capacity interval.** Run `support_projection` without KL. Require a
   predeclared first-order gap tolerance before interpreting its lower bound.
2. **KL reference.** Run the target-space dual `kl_support_projection` with the
   exact prior and effective production KL coefficient when that dual is the
   appropriate dimension. Its primal-dual gap certifies only the fixed-A,
   fixed-offset WLS+KL problem; a non-converged result is unresolved evidence.
3. **Production-q replay.** Freeze the same coordinates and offsets, then evaluate
   the production q under the same KL objective. Compare it with the certified KL
   reference.
4. **Offset controls.** Repeat no-offset and matched bounded-offset arms. Until a
   joint bounded-offset reference solver is implemented, do not call the q-only
   certificate a certificate for the full production optimizer.
5. **Coordinate response.** At fixed q and offsets, compare analytic response with
   central finite differences and with observer values recomputed from emitted
   coordinates. Record data-loss and physical-regularizer gradients separately.
6. **Matched causal controls.** Independently optimize coordinate, no-coordinate,
   no-offset and predictor-only arms with matched seeds, roots and compute budgets.

## Decision rule

| Evidence | Next hypothesis |
| --- | --- |
| WLS interval is wide | Fix the diagnostic solver; infer nothing about support. |
| Certified WLS floor is high and a target-unread new proposal lowers it | Test support discovery/diversity. |
| WLS floor stays high as K grows | Audit observer response and residual-aligned coordinate charts; stop count-only scaling. |
| WLS is low, KL reference is acceptable, production replay is worse | Improve q/offset optimization and conditioning. |
| KL data fit is worse than WLS but production matches KL | Treat as a predeclared regularization tradeoff; change lambda only through source crossfit. |
| Tensor/Jacobian response disagrees with emitted-coordinate recomputation | Repair the coordinate-observer interface before model scaling. |
| Coordinate gain is reproduced by no-coordinate or predictor-only control | Reject the end-to-end generator claim. |

Only one branch selected by a precommitted source-only rule advances. An observed
source residual must not be used to redesign the same fold's supports or thresholds.

## Stability and efficiency

- Cache `A` only while coordinates, observer, normalization and topology are fixed.
- Use FP64 for q, log probabilities, certificates and CCC arithmetic; compare any
  mixed-precision path against that reference.
- Report simplex and logit gradients, ESS/top mass, q sensitivity, solver gap and
  emitted-coordinate displacement together.
- Use matrix-vector products and row streaming at high K. Parallelize independent
  entries/folds with bounded multiprocessing; keep source commitment, once-only
  consumption and final receipt sealing serial.
- Stop the inner solve at a predeclared certificate tolerance tied to demonstrated
  outer-gradient stability, not machine precision by default.
- Pair candidate/control seeds and generation roots. Resample by entry or frozen
  sequence cluster and recompute the original pooled per-Atom_ID CCC in every draw;
  never replace it with an entity-level average.

## Evaluation boundary

The observed shifts of an unseen entry are intentionally used for assimilation and
co-satisfaction scoring. This is not unseen-label prediction. Source diagnostics
may guide a mechanism only through direction-specific crossfit. Validation and the
one-time sealed test remain untouched until a separately committed candidate passes
its source gate. Independent arithmetic checking does not create independent data.
