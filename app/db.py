"""Supabase persistence, against the project schema in supabase/.

Tables used: `scans` (headline verdict per mammogram), `scan_predictions`
(per-architecture breakdown, added by 002_scan_predictions.sql), `profiles`.
Images live in the public `mammograms` Storage bucket under <user_id>/<scan_id>/.

Configured by environment variables (see .env.example):

    SUPABASE_URL=https://<project>.supabase.co
    SUPABASE_SERVICE_KEY=<service_role key>

The SERVICE ROLE key bypasses Row Level Security, so it must never reach the
browser -- every call in this module runs server-side. Ownership is enforced
here instead: each request carries the caller's Supabase access token, which we
verify to get their user id, and every query is scoped to it.

If the variables are absent the API still runs and reports storage as
unconfigured, so the demo works before Supabase is set up.
"""
import base64
import binascii
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
BUCKET = "mammograms"

_client = None
_init_error = None


def is_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_KEY)


def get_client():
    """Lazily build the Supabase client. Returns None when unconfigured."""
    global _client, _init_error
    if _client is not None or not is_configured():
        return _client
    try:
        from supabase import create_client
        _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    except Exception as exc:
        _init_error = str(exc)
        print(f"  supabase: client init failed -- {exc}")
    return _client


def status() -> dict:
    """Reported by /api/health so the UI can show whether records persist."""
    if not is_configured():
        return {"configured": False, "connected": False,
                "detail": "SUPABASE_URL / SUPABASE_SERVICE_KEY not set"}
    client = get_client()
    if client is None:
        return {"configured": True, "connected": False, "detail": _init_error}
    try:
        client.table("scans").select("id").limit(1).execute()
    except Exception as exc:
        return {"configured": True, "connected": False, "detail": str(exc)}

    # scan_predictions is the one table the base schema does not create.
    try:
        client.table("scan_predictions").select("id").limit(1).execute()
        migrated = True
    except Exception:
        migrated = False
    return {"configured": True, "connected": True, "migrated": migrated}


def user_from_token(access_token: str):
    """Verify a Supabase access token and return its user id, or None."""
    client = get_client()
    if client is None or not access_token:
        return None
    try:
        res = client.auth.get_user(access_token)
        return res.user.id if res and res.user else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------
def _decode_data_uri(data_uri: str) -> bytes:
    if not data_uri or "," not in data_uri:
        raise ValueError("not a data URI")
    try:
        return base64.b64decode(data_uri.split(",", 1)[1])
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"malformed base64 payload: {exc}") from exc


def upload_png(user_id: str, scan_id: str, filename: str, data_uri: str) -> str:
    """Put one PNG in the bucket under <user_id>/<scan_id>/ and return its public URL."""
    client = get_client()
    path = f"{user_id}/{scan_id}/{filename}"
    client.storage.from_(BUCKET).upload(
        path,
        _decode_data_uri(data_uri),
        {"content-type": "image/png", "upsert": "true"},
    )
    return client.storage.from_(BUCKET).get_public_url(path)


def _remove_scan_files(user_id: str, scan_id: str) -> None:
    """Best-effort cleanup of a scan's folder; never blocks the row delete."""
    client = get_client()
    try:
        prefix = f"{user_id}/{scan_id}"
        entries = client.storage.from_(BUCKET).list(prefix)
        paths = [f"{prefix}/{e['name']}" for e in entries]
        if paths:
            client.storage.from_(BUCKET).remove(paths)
    except Exception as exc:
        print(f"  supabase: storage cleanup skipped for {scan_id} -- {exc}")


# ---------------------------------------------------------------------------
# row <-> API shape
# ---------------------------------------------------------------------------
def _pred_to_model(p):
    return {
        "arch": p["arch"],
        "displayName": p["display_name"],
        "probability": float(p["probability"]),
        "threshold": float(p["threshold"]),
        "result": p["result"],
        "confidence": float(p["confidence"]),
        "thresholdVariant": p.get("threshold_variant"),
        "isPrimary": p.get("is_primary", False),
        "heatmap": p.get("heatmap_url"),
    }


def _row_to_record(row, preds=None):
    """Database row -> the record shape the frontend renders."""
    preds = preds or []
    models = [_pred_to_model(p) for p in preds]
    primary = next((m for m in models if m["isPrimary"]), models[0] if models else None)

    created = (row.get("created_at") or "")[:10]
    analysis = None
    if row.get("image_url") or models:
        analysis = {
            "primaryArch": primary["arch"] if primary else "densenet121",
            "device": None,
            "originalImage": row.get("image_url"),
            "probability": primary["probability"] if primary else 0.0,
            "threshold": primary["threshold"] if primary else 0.5,
            "result": row["result"],
            "confidence": float(row.get("confidence_score") or 0),
            "models": models,
            "groundTruth": (
                {"available": True, "label": row["ground_truth"], "roiImage": row.get("roi_url")}
                if row.get("ground_truth") else None
            ),
        }

    return {
        "id": row["id"],
        "name": row["image_name"],
        "date": created,
        "caseDate": created,
        "result": row["result"],
        "confidence": float(row.get("confidence_score") or 0),
        "patientName": row.get("patient_name"),
        "patientAge": "—",
        "densityGrade": "—",
        "radiologist": "—",
        "protocol": "Uploaded scan",
        "summary": _summary_for(row, primary),
        "notes": row.get("clinical_notes") or "",
        "notesEdited": (row.get("updated_at") or "")[:19].replace("T", " "),
        "analysis": analysis,
    }


def _summary_for(row, primary):
    if not primary:
        return ""
    name, p, t = primary["displayName"], primary["probability"], primary["threshold"]
    if row["result"] == "Malignant":
        return (f"{name} flags this study as malignant at p={p:.3f} "
                f"(threshold {t:.3f}). Radiologist review required.")
    return f"{name} returns p(malignant)={p:.3f}, below the {t:.3f} decision threshold."


# ---------------------------------------------------------------------------
# CRUD -- every call is scoped to user_id, so the service key cannot be used
# to read another clinician's scans.
# ---------------------------------------------------------------------------
def list_records(user_id):
    client = get_client()
    if client is None:
        return None
    # No predictions joined here -- the archive table shows no per-model detail.
    res = (client.table("scans").select("*")
           .eq("user_id", user_id).order("created_at", desc=True).execute())
    return [_row_to_record(r) for r in res.data]


def get_record(user_id, scan_id):
    client = get_client()
    if client is None:
        return None
    res = (client.table("scans").select("*")
           .eq("id", scan_id).eq("user_id", user_id).limit(1).execute())
    if not res.data:
        return None
    preds = (client.table("scan_predictions").select("*")
             .eq("scan_id", scan_id).order("is_primary", desc=True).execute())
    return _row_to_record(res.data[0], preds.data)


def save_analysis(user_id, image_name, analysis, patient_name=None):
    """Insert a scan, upload its images, and record the per-architecture breakdown."""
    client = get_client()
    if client is None:
        return None

    primary = next(m for m in analysis["models"] if m["arch"] == analysis["primaryArch"])
    row = {
        "user_id": user_id,
        "image_name": image_name,
        "patient_name": patient_name,
        "image_url": "",             # filled in below, once we know the scan id
        "result": analysis["result"],
        "confidence_score": analysis["confidence"],
        "clinical_notes": "",
    }
    scan_id = client.table("scans").insert(row).execute().data[0]["id"]

    try:
        patch = {"image_url": upload_png(user_id, scan_id, "original.png",
                                         analysis["originalImage"])}
        gt = analysis.get("groundTruth")
        if gt and gt.get("roiImage"):
            patch["roi_url"] = upload_png(user_id, scan_id, "roi.png", gt["roiImage"])
            patch["ground_truth"] = gt["label"]

        preds = []
        for m in analysis["models"]:
            url = upload_png(user_id, scan_id, f"heatmap_{m['arch']}.png", m["heatmap"])
            if m["arch"] == primary["arch"]:
                patch["heatmap_url"] = url
            preds.append({
                "scan_id": scan_id,
                "arch": m["arch"],
                "display_name": m["displayName"],
                "probability": m["probability"],
                "threshold": m["threshold"],
                "result": m["result"],
                "confidence": m["confidence"],
                "threshold_variant": m.get("thresholdVariant"),
                "is_primary": m["arch"] == primary["arch"],
                "heatmap_url": url,
            })

        client.table("scans").update(patch).eq("id", scan_id).execute()
        client.table("scan_predictions").insert(preds).execute()
    except Exception:
        # Never leave a half-written scan behind for the UI to render.
        client.table("scans").delete().eq("id", scan_id).execute()
        _remove_scan_files(user_id, scan_id)
        raise

    return get_record(user_id, scan_id)


def update_notes(user_id, scan_id, notes):
    client = get_client()
    if client is None:
        return None
    stamp = datetime.now(timezone.utc).isoformat()
    res = (client.table("scans")
           .update({"clinical_notes": notes, "updated_at": stamp})
           .eq("id", scan_id).eq("user_id", user_id).execute())
    return bool(res.data)


def delete_record(user_id, scan_id):
    client = get_client()
    if client is None:
        return None
    # scan_predictions rows go with it via ON DELETE CASCADE.
    res = (client.table("scans").delete()
           .eq("id", scan_id).eq("user_id", user_id).execute())
    if not res.data:
        return False
    _remove_scan_files(user_id, scan_id)
    return True


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------
def get_profile(user_id):
    client = get_client()
    if client is None:
        return None
    res = client.table("profiles").select("*").eq("id", user_id).limit(1).execute()
    if not res.data:
        return None
    r = res.data[0]
    return {
        "fullName": r.get("full_name") or "",
        "email": r.get("email") or "",
        "role": r.get("clinical_role") or "",
        "medicalId": r.get("medical_id") or "",
    }


def update_profile(user_id, patch):
    client = get_client()
    if client is None:
        return None
    row = {}
    for src, dst in (("fullName", "full_name"), ("role", "clinical_role")):
        if src in patch:
            row[dst] = patch[src]
    # email is owned by auth.users and medical_id by hospital administration --
    # neither is editable from the settings form.
    if not row:
        return get_profile(user_id)
    client.table("profiles").update(row).eq("id", user_id).execute()
    return get_profile(user_id)
