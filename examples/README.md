# Precomputed example

`book_1/gpt-4o/` contains one real API run for each of the three inference
modes: Full-book E2E, CoT, and MOSAIC. For each mode, the directory includes
the raw prediction, the normalized prediction produced by the post-processing
code, and the resulting evaluation summary.

These files are a compact, no-API reference for checking the expected formats
and the complete data flow. They are illustrative single-run outputs for one
released book, not the aggregate results reported in the paper. Model APIs are
nondeterministic and service versions can change, so a new run need not match
these files exactly.

Re-run post-processing and evaluation for all three predictions from the
repository root with:

```bash
USE_PRECOMPUTED=1 MODE=all bash scripts/run_pipeline.sh
```
