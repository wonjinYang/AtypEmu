# Job 134710: invalidated K=8 dynamic-coordinate source gate

Job `134710` is **not** a valid source-gate pass. The frozen plan required label eligibility to be applied before source-fold normalization, observer fitting, and assimilation; the executed runner applied it only during scoring. Its positive values and checker arithmetic `PASS` are non-authoritative quarantine diagnostics and must not be used for selection, promotion, validation, or authorization. They do not replace retained Run 98.

No outer or formal metric was opened. The authorization was consumed and cannot be reused.

Historical `.auto/log.jsonl` rows 7 and 8 are retained unchanged for auditability but are superseded by `log_supersession.json`; neither row is valid model-selection evidence.

`post_handoff_audit.json` is the canonical promotion disposition. Exact executed code and compact receipts are preserved at commit `95a8906` on branch `quarantine/job134710-invalid-source-gate`. The 13 remote recovery files, including all six hashed outputs, were restored and hash-verified under `.auto/consolidation/job134710_20260903/`; that ignored local archive is evidence only. The previously referenced K=32 WIP tarball is absent from the recovered local archive after cleanup and is not reconstructed because those unverified bytes are forbidden to resume. The accidental logger commit is preserved separately on `quarantine/accidental-log-tool-badb416`.

For this mechanism, the only eligible continuation is a separately named corrected K=8 job with a new frozen commitment, independent pre-execution review, and new external once-only authorization. Nothing from Job 134710's authorization or scored result may be reused as approval. The existing K=32 WIP must not be resumed, repaired in place, executed, or promoted; any future K=32 attempt requires a new valid parent gate and fresh authorization.

`gpuopt/preflight_all_label_autoresearch.py` is a setup-only sanity check, not authorization evidence. Before any corrected K=8 authorization, an independent checker must recompute source-fold eligible labels and exact eligible row identities from frozen raw source bytes, then prove that the same filtered identities—and no excluded identities—enter normalization, observer fitting, and assimilation.
