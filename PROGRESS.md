# Progress (autonomous run; started 2026-11-16 ~01:05)

Scope per user: CNN (A) vs MERT (C). AST skipped (code is pluggable; configs/ast.yaml is a placeholder).

- [x] M1 Environment: .venv (py3.12, CUDA torch, transformers<5), metadata + fma_small extracted, fma_medium download started
- [x] M1b fma_medium downloaded, SHA1 OK, extracted (25,000 mp3)
- [x] M2 splits + fma_small cache (7,994/8,000 ok, 6 bad) + sample plot checked (outputs/figures/sample_small.png)
- [x] M3 CNN end-to-end on fma_small: smoke OK, 3 epochs val macro-F1 0.44, evaluate.py works (logs/cnn_quick.log)
- [x] M4 Full cache on fma_medium: 24,980 of 25,000 tracks cached in 43 min; 20 bad tracks logged (data/bad_tracks.txt)
- [x] M5 Full CNN trained on fma_medium: best val macro-F1 0.4635 (ep 26); TEST acc 0.671, top-3 0.845, macro-F1 0.460 (outputs/cnn)
- [x] M6 MERT: embeddings extracted (24,980), head trained: TEST acc 0.682, top-3 0.846, macro-F1 0.493 (outputs/mert)
- [-] M7 AST: skipped by user decision (days 3-4); registry + placeholder config ready
- [x] M8 Evaluation report with comparison, confusion matrices, error analysis: reports/results.md
- [x] M9 Gradio demo tested live via API (auto-picks mert); microphone confirmed working by the user
- [x] M10 README.md

## Overnight chain launched
`scripts/run_overnight.sh` (log: logs/overnight.log) runs M4 -> M5 -> M6 -> final eval. Re-run it if it stops; each step is resumable.

- [x] Todo item 1 (imbalance + seeds + CIs): done 2026-11-17, see reports/imbalance_and_uncertainty.md

## Autonomous run 2 (started 2026-11-17): user asked for CNN seeds, then housekeeping, error-analysis tests, AST
- [x] T1 CNN extra seeds done (4 seeds: test macro-F1 0.465 ± 0.005, acc 0.673 ± 0.002); MERT-CNN gap +0.031, 95% CI [0.010, 0.050] (reports/imbalance_and_uncertainty.md)
- [x] T2 Housekeeping done: quick runs + debug logs deleted; zips intentionally kept (see decisions)
- [x] T3 Error-analysis tests done: H1,H3,H4,H5 supported/consistent, H2 not testable (single-genre subset); see reports/error_analysis_tests.md
- [x] T4 AST done: reduced (8 layers frozen) test acc 0.708 / macro-F1 0.538; full fine-tune acc 0.702 / 0.531; demo tested; docs updated (reports/results.md, README)
