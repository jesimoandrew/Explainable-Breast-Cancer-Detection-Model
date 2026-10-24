"""SQLite stand-in for app/db.py, for local testing without Supabase.

Selected by ASTRA_DB=sqlite (see .env.sqlite / `npm run dev:local` in frontend/).
Exposes the same functions as app/db.py, so app/server.py is unchanged apart from
choosing which module to import. Everything lives under local-data/ (gitignored):

    local-data/astrascan.db                 scans, scan_predictions, profiles, users, sessions
    local-data/files/<user>/<scan>/*.png    images, served by the API at /api/files/...

Auth is local too: /api/auth/signup and /api/auth/login (app/server.py) issue an
opaque bearer token stored in `sessions`. It is for local testing only -- there
is no email confirmation, rate limiting or token expiry.
"""
import hashlib
import hmac
import os
import secrets
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.db import _decode_data_uri, _row_to_record

DATA_DIR = Path(os.getenv("ASTRA_LOCAL_DATA",
                          Path(__file__).resolve().parent.parent / "local-data"))
FILES_DIR = DATA_DIR / "files"
DB_PATH = DATA_DIR / "astrascan.db"
FILES_URL = "/api/files"   # same-origin through the Vite proxy

SCHEMA = """
create table if not exists users (
  id text primary key,
  email text not null unique,
  password_hash text not null,
  created_at text not null
);
create table if not exists sessions (
  token text primary key,
  user_id text not null references users(id) on delete cascade,
  created_at text not null
);
create table if not exists profiles (
  id text primary key references users(id) on delete cascade,
  full_name text,
  email text,
  clinical_role text,
  medical_id text
);
create table if not exists scans (
  id text primary key,
  user_id text not null references users(id) on delete cascade,
  image_name text not null,
  patient_name text,
  image_url text,
  roi_url text,
  heatmap_url text,
  ground_truth text check (ground_truth is null or ground_truth in ('Benign', 'Malignant')),
  result text not null check (result in ('Benign', 'Malignant')),
  confidence_score real,
  clinical_notes text default '',
  created_at text not null,
  updated_at text not null
);
create index if not exists scans_user_idx on scans(user_id, created_at);
create table if not exists scan_predictions (
  id text primary key,
  scan_id text not null references scans(id) on delete cascade,
  arch text not null,
  display_name text not null,
  probability real not null,
  calibrated_probability real,
  threshold real not null,
  result text not null check (result in ('Benign', 'Malignant')),
  confidence real not null,
  threshold_variant text,
  is_primary integer not null default 0,
  heatmap_url text,
  created_at text not null,
  unique (scan_id, arch)
);
"""

BUCKET = "mammograms"   # kept for parity with app/db.py


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("pragma foreign_keys = on")
    conn.execute("pragma journal_mode = wal")
    return conn


def _init() -> None:
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(SCHEMA)


_init()


def is_configured() -> bool:
    return True


def status() -> dict:
    try:
        with _connect() as conn:
            conn.execute("select 1 from scans limit 1")
    except Exception as exc:
        return {"configured": True, "connected": False, "backend": "sqlite", "detail": str(exc)}
    return {"configured": True, "connected": True, "migrated": True, "backend": "sqlite"}


# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------
def _hash_password(password: str, salt: bytes = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1)
    return f"{salt.hex()}${digest.hex()}"


def _check_password(password: str, stored: str) -> bool:
    salt_hex, digest_hex = stored.split("$", 1)
    candidate = _hash_password(password, bytes.fromhex(salt_hex)).split("$", 1)[1]
    return hmac.compare_digest(candidate, digest_hex)


def _new_session(conn, user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    conn.execute("insert into sessions (token, user_id, created_at) values (?, ?, ?)",
                 (token, user_id, _now()))
    return token


def sign_up(email: str, password: str, full_name: str = ""):
    """Create a user + profile. Returns the user id, or None if the email is taken."""
    email = email.strip().lower()
    user_id = str(uuid.uuid4())
    try:
        with _connect() as conn:
            conn.execute("insert into users (id, email, password_hash, created_at) "
                         "values (?, ?, ?, ?)",
                         (user_id, email, _hash_password(password), _now()))
            conn.execute("insert into profiles (id, full_name, email) values (?, ?, ?)",
                         (user_id, full_name.strip(), email))
    except sqlite3.IntegrityError:
        return None
    return user_id


def sign_in(email: str, password: str):
    """Returns {accessToken, user: {id, email}} or None for bad credentials."""
    email = email.strip().lower()
    with _connect() as conn:
        user = conn.execute("select * from users where email = ?", (email,)).fetchone()
        if not user or not _check_password(password, user["password_hash"]):
            return None
        token = _new_session(conn, user["id"])
    return {"accessToken": token, "user": {"id": user["id"], "email": user["email"]}}


def sign_out(token: str) -> None:
    with _connect() as conn:
        conn.execute("delete from sessions where token = ?", (token,))


def user_from_token(access_token: str):
    if not access_token:
        return None
    with _connect() as conn:
        row = conn.execute("select user_id from sessions where token = ?",
                           (access_token,)).fetchone()
    return row["user_id"] if row else None


def user_info(user_id: str):
    with _connect() as conn:
        row = conn.execute("select id, email from users where id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------
def upload_png(user_id: str, scan_id: str, filename: str, data_uri: str) -> str:
    folder = FILES_DIR / user_id / scan_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / filename).write_bytes(_decode_data_uri(data_uri))
    return f"{FILES_URL}/{user_id}/{scan_id}/{filename}"


def _remove_scan_files(user_id: str, scan_id: str) -> None:
    shutil.rmtree(FILES_DIR / user_id / scan_id, ignore_errors=True)


# ---------------------------------------------------------------------------
# CRUD -- every call is scoped to user_id, as in app/db.py
# ---------------------------------------------------------------------------
def list_records(user_id):
    with _connect() as conn:
        rows = conn.execute("select * from scans where user_id = ? order by created_at desc",
                            (user_id,)).fetchall()
    return [_row_to_record(dict(r)) for r in rows]


def get_record(user_id, scan_id):
    with _connect() as conn:
        row = conn.execute("select * from scans where id = ? and user_id = ?",
                           (scan_id, user_id)).fetchone()
        if not row:
            return None
        preds = conn.execute("select * from scan_predictions where scan_id = ? "
                             "order by is_primary desc", (scan_id,)).fetchall()
    preds = [{**dict(p), "is_primary": bool(p["is_primary"])} for p in preds]
    return _row_to_record(dict(row), preds)


def save_analysis(user_id, image_name, analysis, patient_name=None):
    primary = next(m for m in analysis["models"] if m["arch"] == analysis["primaryArch"])
    scan_id = str(uuid.uuid4())
    now = _now()

    try:
        gt = analysis.get("groundTruth")
        image_url = upload_png(user_id, scan_id, "original.png", analysis["originalImage"])
        roi_url = ground_truth = None
        if gt and gt.get("roiImage"):
            roi_url = upload_png(user_id, scan_id, "roi.png", gt["roiImage"])
            ground_truth = gt["label"]

        preds, primary_heatmap = [], None
        for m in analysis["models"]:
            url = upload_png(user_id, scan_id, f"heatmap_{m['arch']}.png", m["heatmap"])
            is_primary = m["arch"] == primary["arch"]
            if is_primary:
                primary_heatmap = url
            preds.append((str(uuid.uuid4()), scan_id, m["arch"], m["displayName"],
                          m["probability"], m["calibratedProbability"], m["threshold"],
                          m["result"], m["confidence"], m.get("thresholdVariant"),
                          int(is_primary), url, now))

        # One transaction: a failure leaves no half-written scan behind.
        with _connect() as conn:
            conn.execute(
                "insert into scans (id, user_id, image_name, patient_name, image_url, "
                "roi_url, heatmap_url, ground_truth, result, confidence_score, "
                "clinical_notes, created_at, updated_at) "
                "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?, ?)",
                (scan_id, user_id, image_name, patient_name, image_url, roi_url,
                 primary_heatmap, ground_truth, analysis["result"],
                 analysis["confidence"], now, now))
            conn.executemany(
                "insert into scan_predictions (id, scan_id, arch, display_name, "
                "probability, calibrated_probability, threshold, result, confidence, "
                "threshold_variant, is_primary, heatmap_url, created_at) "
                "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", preds)
    except Exception:
        _remove_scan_files(user_id, scan_id)
        raise

    return get_record(user_id, scan_id)


def update_notes(user_id, scan_id, notes):
    with _connect() as conn:
        cur = conn.execute("update scans set clinical_notes = ?, updated_at = ? "
                           "where id = ? and user_id = ?",
                           (notes, _now(), scan_id, user_id))
    return cur.rowcount > 0


def delete_record(user_id, scan_id):
    with _connect() as conn:   # scan_predictions cascade via the foreign key
        cur = conn.execute("delete from scans where id = ? and user_id = ?",
                           (scan_id, user_id))
    if cur.rowcount == 0:
        return False
    _remove_scan_files(user_id, scan_id)
    return True


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------
def get_profile(user_id):
    with _connect() as conn:
        r = conn.execute("select * from profiles where id = ?", (user_id,)).fetchone()
    if not r:
        return None
    return {
        "fullName": r["full_name"] or "",
        "email": r["email"] or "",
        "role": r["clinical_role"] or "",
        "medicalId": r["medical_id"] or "",
    }


def update_profile(user_id, patch):
    row = {}
    for src, dst in (("fullName", "full_name"), ("role", "clinical_role")):
        if src in patch:
            row[dst] = patch[src]
    if row:
        sets = ", ".join(f"{col} = ?" for col in row)
        with _connect() as conn:
            conn.execute(f"update profiles set {sets} where id = ?",
                         (*row.values(), user_id))
    return get_profile(user_id)
