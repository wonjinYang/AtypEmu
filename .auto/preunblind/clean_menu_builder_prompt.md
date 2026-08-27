# Clean-context Phase-D candidate-menu recovery

You are constructing a finite candidate menu, not selecting a winner and not
assessing any outer result. You have no outer-result files or metrics. Use only
the evidence below. If it is insufficient for an executable, leakage-safe menu,
return `NO_SAFE_MENU` and explain the missing precondition.

## Governing user compass

The separately supplied 1,404-byte user message has SHA-256
`a52f05b66ac3fe97925dc7e532c1f8556322b5546c769120ec3643a9cf86fc13`.
Treat it as scientific direction whose temporal priority still needs
corroboration; do not treat it as external truth.

## Dated pre-unblinding development records

These records are from `autoresearch.jsonl` runs 1--20, all dated
2026-05-19, months before the current Phase-D evaluation. They were committed
as the experiment chain and contain the following exact candidate precedents:

- run 7, commit `610e32f`: `SER:HN:exchange:Q4`, +0.02 half-step. Four
  crossfit folds had positive mean but one negative holdout fold. It was later
  completed to +0.04 and marked at its old configured maximum.
- runs 13--14, commits `2df1e7c` and `fb6c676`:
  `GLY:HN:sulfur_electrostatic:Q1`, +0.005 increments. Historical crossfit was
  mixed; the old chart reached +0.04 and was marked complete.
- run 7's dated next-action record names a distinct
  `PHE:HN:sulfur_electrostatic:Q4` route as an untried alternative.
- runs 10--11, commits `88b54fc` and `ec0c319`:
  `THR:HN:ring_current:Q4`, +0.005 increments. Crossfit entity direction was
  negative and the old chart reached its configured maximum.
- run 2, commit `694f3af`: `GLU:HN:alignment_mismatch:Q1`, +0.01 increment,
  with positive crossfit entity-macro evidence. The current Phase-D physical
  observer has no explicit alignment-mismatch feature.

These are precedents from an older chart architecture. Do not assume that an
old positive result transfers to Phase D, and do not add another correction to
an old maxed chart unless the new mechanism is explicitly distinct and a
train-to-inner-dev counterfactual independently supports it.

## Current Phase-D observation surface (fixed before evaluation)

Source SHA-256:
`0b3e6dc0974e0da9387cfc7cf5113c46c47b88fd85d2373be812e1020f397a41`.

- Every support conformer has exact atom27 coordinates and seven boolean atom
  roles: `H`, `C`, `N`, `O`, `S`, `donor`, `acceptor`.
- The observer already computes smooth role-distance RBFs at centers
  `(0.20, 0.30, 0.40, 0.55, 0.75, 1.00)` nm with width `0.12` nm, separately
  for same-residue and inter-residue atoms. Its normalization is
  `sum(rbf) / sqrt(max(role_count, 1))`.
- For each support conformer and target, the observer produces a support mean.
  The ensemble prediction is `sum_k(q_k * support_mean_k)`. A deterministic
  observation-chart correction may be added to support means after `q` is
  computed; this leaves posterior weights, ESS, and entropy unchanged.
- Target metadata identifies HN and the target residue name. Exact geometry
  identifies sulfur and acceptor environments. There is no explicit aromatic
  ring-current scalar, exchange-state label, terminal-state label, or
  alignment-mismatch label in the frozen batch.

## Menu rules

1. Admit only a residue/family/mechanism precedent named above and a feature
   already available on the fixed Phase-D surface.
2. Do not invent a candidate from any current result.
3. Every candidate must specify an exact formula, support gate, sign set,
   magnitude, and deterministic ID. Both signs may be enumerated before seeing
   data if the old chart sign is not semantically transferable.
4. Use a smooth feature in `[0,1]`; add at most 0.02 ppm to an HN support mean.
5. Enumerating the six pre-existing RBF centers is allowed. Do not tune a new
   center or width.
6. Exclude ring, terminal, or alignment routes if their mechanism cannot be
   calculated from the fixed surface without a new representation.
7. Selection later uses only inner-train and inner-dev folds. It must require
   same-sign improvement in all three folds, no HN raw/standardized p95 or max
   worsening, lower Student-t objective, and exact invariance of C', CA, CB, N,
   and posterior q. No candidate passing means `NO_CANDIDATE`.

Return one JSON object with keys `decision`, `candidate_schema`, `candidates`,
`exclusions`, and `leakage_notes`. Prefer the smallest defensible menu.
