# AstraScan AI — Frontend

Clinician-facing UI for the explainable breast-cancer detection system. Plain
JavaScript React (no TypeScript), Vite, React Router. No UI framework — all
styling lives in a single stylesheet driven by CSS custom properties.

Uploads run real inference against the four trained hi-res architectures.

## Run

One command starts both the inference API and the UI:

```bash
cd frontend
npm install      # first time only
npm run dev:all  # http://localhost:5173
```

Output is tagged `[api]` and `[ui]`. Ctrl+C stops both. The API needs ~10s to
load the four models, so the UI will briefly report the engine as offline and log
one `ECONNREFUSED` proxy error — it clears itself once `Ready on cuda` appears.

To run them separately instead:

```bash
npm run dev:api  # inference API on :8000 (uv run uvicorn app.server:app)
npm run dev      # UI on :5173
```

Vite proxies `/api/*` to `127.0.0.1:8000`, so the browser only sees one origin.

## Models behind the API

| Architecture | Checkpoint | Threshold |
| --- | --- | --- |
| **DenseNet121** (primary) | `notebooks/experiment_2_optimization/checkpoints/hires_control_gap_a0.5_densenet121_high_sensitivity_t0.305.pth` | **0.305** |
| ResNet-50 | `models/v4_hires_control_gap/hires_control_gap_a0.5_resnet50.pth` | 0.500 |
| EfficientNet-B2 | `models/v4_hires_control_gap/hires_control_gap_a0.5_efficientnet_b2.pth` | 0.500 |
| VGG-16 | `models/v4_hires_control_gap/hires_control_gap_a0.5_vgg16.pth` | 0.500 |

The three plain checkpoints are the hi-res models from
`notebooks/experiment_1_baseline/hires_architecture_comparison.ipynb`. DenseNet121
is served from the sensitivity-optimized variant, whose 0.305 cutoff is read from
the checkpoint's own `decision_threshold` field rather than hard-coded. It drives
the headline verdict; the other three appear in the Architecture Comparison panel.

## Supabase

Required — sign-in and the scan archive both run on it.

1. **Run the migration.** Paste
   [`supabase/002_scan_predictions.sql`](../supabase/002_scan_predictions.sql)
   into the SQL Editor and run it. Additive and idempotent: it adds
   `scan_predictions` (one row per architecture per scan) plus `roi_url` /
   `ground_truth` on `scans`. Nothing existing is modified.
2. **Backend key** — project-root `.env`:
   ```
   SUPABASE_URL=https://<project-ref>.supabase.co
   SUPABASE_SERVICE_KEY=<service_role key>
   ```
3. **Browser key** — `frontend/.env`:
   ```
   VITE_SUPABASE_URL=https://<project-ref>.supabase.co
   VITE_SUPABASE_ANON_KEY=<anon public key>
   ```

### Which key goes where

The **anon** key ships in the browser bundle by design and is constrained by RLS.
The **service_role** key bypasses RLS entirely and is read only by
[`app/db.py`](../app/db.py) — it must never appear in `frontend/`. Both `.env`
files are gitignored. If the service key is ever committed or pasted into client
code, rotate it in Project Settings → API.

Because the backend holds a key that bypasses RLS, it cannot rely on RLS for
isolation. Instead each request carries the caller's Supabase access token; the
API verifies it and scopes every query to that user id, so one clinician cannot
read another's scans.

### Data layout

| Table | Holds |
| --- | --- |
| `profiles` | clinician identity, driven by the signup trigger |
| `scans` | one row per mammogram — headline verdict from DenseNet121 |
| `scan_predictions` | four rows per scan — probability, threshold, verdict, heatmap URL |

Images go to the public `mammograms` bucket under `<user_id>/<scan_id>/`:
`original.png`, `heatmap_<arch>.png` ×4, and `roi.png` when a ground-truth
annotation matches. Deleting a scan cascades the predictions and clears the folder.

## Endpoints

| Endpoint | Returns |
| --- | --- |
| `GET /api/health` | status, device, models loaded, storage state |
| `GET /api/models` | per-architecture checkpoint + threshold metadata |
| `POST /api/analyze` | verdict, p(malignant), Grad-CAM per architecture; saves if configured |
| `GET /api/records` | archive without image payloads |
| `GET /api/records/{id}` | one record including images |
| `PATCH /api/records/{id}` | update clinical notes |
| `DELETE /api/records/{id}` | delete record and its images |

## Pages and routes

| Route              | Page                | Notes                                             |
| ------------------ | ------------------- | ------------------------------------------------- |
| `/login`           | Login               | No header/nav. Any credentials sign you in.        |
| `/dashboard`       | Dashboard           | Welcome hero + 3 most recent scans                 |
| `/records`         | Patient Records     | Full archive, searchable and filterable            |
| `/records/:id`     | Record Details      | Scan + Grad-CAM panels, classification, notes      |
| `/analysis`        | AI Analysis         | Drag & drop upload, creates a new record           |
| `/settings/:tab`   | Settings            | `profile` \| `security` \| `preferences`           |

`/` and any unknown path redirect to `/dashboard`. Every route except `/login`
is behind an auth guard that bounces you to the login screen and returns you to
the page you asked for after sign-in.

## Working navigation

- Header nav links to all four sections; the active item is highlighted.
- Brand logo → dashboard. Logout → clears auth, returns to `/login`.
- Dashboard "Analyze Image" → `/analysis`; "View All Records" → `/records`.
- Any table row (or its eye icon) → that record's detail page.
- Record Details back arrow → previous page; Delete → confirms, removes, returns
  to `/records`; Save Changes → persists the clinical notes.
- AI Analysis "Upload & Analyze" → creates a record and opens its detail page.
- Settings sidebar items are real routes, so they are linkable and survive reload.

## Structure

```
src/
  main.jsx              app entry, router + provider
  App.jsx               route table and auth guard
  styles.css            design tokens and all component styles
  context/AppContext    auth, clinician profile, record store
  data/records.js       seed records
  components/           Header, Footer, Layout, RecordsTable, Mammogram, Icons
  pages/                Login, Dashboard, PatientRecords, RecordDetails,
                        AiAnalysis, Settings
```

## Prototype boundaries

Case-summary fields the model does not produce (patient age, density grade,
radiologist) render as `—` rather than being invented. `scans` has a
`patient_name` column that nothing writes yet.

Scans created by the earlier version of this app predate `scan_predictions`, so
they show a headline verdict but an empty Architecture Comparison panel. Re-run
those images to populate it.

`POST /api/analyze` takes PNG/JPEG/BMP/TIFF. The upload copy still advertises
DICOM, but the endpoint does not decode `.dcm` yet — `pydicom` is already a
project dependency, so that is a small addition to `app/server.py`.

Password reset and Hospital ID SSO are stubs that show a notice.

All scan frames share one aspect ratio via the `--scan-aspect` token in
`styles.css` (default `3 / 4`, portrait to suit mammograms). Change that single
value to reshape every frame — panels, comparison tiles, ROI, and thumbnails.

`POST /api/analyze` takes PNG/JPEG/BMP/TIFF. The upload copy still advertises
DICOM, but the endpoint does not decode `.dcm` yet — `pydicom` is already a
project dependency, so that is a small addition to `app/server.py` when needed.
