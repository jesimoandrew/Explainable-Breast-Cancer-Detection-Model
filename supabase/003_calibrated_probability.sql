-- ============================================================
-- AstraScan AI -- migration 003: calibrated probability
-- Run in Supabase Dashboard -> SQL Editor, after 002.
-- Additive only. Idempotent -- safe to re-run.
-- ============================================================
--
-- Each model's p(malignant) is now also reported after temperature scaling
-- (notebooks/experiment_2_optimization/confidence_calibration.ipynb). The
-- decision still uses the raw `probability` against `threshold`; `confidence`
-- (and scans.confidence_score) now hold the calibrated confidence in the
-- predicted class. Rows saved before this migration keep a null here.

alter table scan_predictions
  add column if not exists calibrated_probability numeric(6,5);
