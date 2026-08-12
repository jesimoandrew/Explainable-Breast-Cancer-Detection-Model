-- ============================================================
-- AstraScan AI -- migration 002: per-architecture predictions
-- Run in Supabase Dashboard -> SQL Editor, after the base schema.
-- Additive only: `scans`, `profiles`, and storage are untouched.
-- Idempotent -- safe to re-run.
-- ============================================================
--
-- Each scan is classified by four hi-res architectures (DenseNet121,
-- ResNet-50, EfficientNet-B2, VGG-16), each producing its own probability,
-- decision threshold and Grad-CAM overlay. `scans` holds the headline verdict
-- from the primary model (DenseNet121, threshold 0.305); this table holds the
-- full breakdown behind it.

create table if not exists scan_predictions (
  id uuid primary key default gen_random_uuid(),
  scan_id uuid not null references scans(id) on delete cascade,

  arch text not null,                 -- densenet121 | resnet50 | efficientnet_b2 | vgg16
  display_name text not null,         -- "DenseNet121", "ResNet-50", ...
  probability numeric(6,5) not null,  -- raw p(malignant)
  threshold numeric(6,5) not null,    -- cutoff applied (0.305 for the optimized DenseNet121)
  result text not null check (result in ('Benign', 'Malignant')),
  confidence numeric(5,2) not null,   -- confidence in the predicted class, 0-100
  threshold_variant text,             -- e.g. 'high_sensitivity'; null for the plain checkpoints
  is_primary boolean not null default false,
  heatmap_url text,                   -- Grad-CAM overlay in the mammograms bucket
  created_at timestamptz default now(),

  unique (scan_id, arch)
);

create index if not exists scan_predictions_scan_id_idx on scan_predictions(scan_id);

alter table scan_predictions enable row level security;

-- Ownership is inherited from the parent scan, matching the policies on `scans`.
drop policy if exists "Users can view own scan predictions" on scan_predictions;
create policy "Users can view own scan predictions"
  on scan_predictions for select
  using (exists (
    select 1 from scans s where s.id = scan_predictions.scan_id and s.user_id = auth.uid()
  ));

drop policy if exists "Users can insert own scan predictions" on scan_predictions;
create policy "Users can insert own scan predictions"
  on scan_predictions for insert
  with check (exists (
    select 1 from scans s where s.id = scan_predictions.scan_id and s.user_id = auth.uid()
  ));

drop policy if exists "Users can update own scan predictions" on scan_predictions;
create policy "Users can update own scan predictions"
  on scan_predictions for update
  using (exists (
    select 1 from scans s where s.id = scan_predictions.scan_id and s.user_id = auth.uid()
  ));

drop policy if exists "Users can delete own scan predictions" on scan_predictions;
create policy "Users can delete own scan predictions"
  on scan_predictions for delete
  using (exists (
    select 1 from scans s where s.id = scan_predictions.scan_id and s.user_id = auth.uid()
  ));

-- ------------------------------------------------------------
-- Ground-truth ROI overlay, shown for comparison when an uploaded filename
-- matches a CBIS-DDSM row. Nullable: externally supplied scans have none.
-- ------------------------------------------------------------
alter table scans add column if not exists roi_url text;
alter table scans add column if not exists ground_truth text;

do $$
begin
  if not exists (
    select 1 from pg_constraint where conname = 'scans_ground_truth_check'
  ) then
    alter table scans add constraint scans_ground_truth_check
      check (ground_truth is null or ground_truth in ('Benign', 'Malignant'));
  end if;
end $$;
