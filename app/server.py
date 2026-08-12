"""HTTP inference API behind the AstraScan AI frontend.

Wraps app/inference.py -- the same path the Streamlit demo in app/app.py uses --
so the React app in frontend/ can run real classifications against the four
trained hi-res architectures, and app/db.py so results persist to Supabase.

Auth: the browser signs in with Supabase Auth (anon key) and sends its access
token as `Authorization: Bearer <token>`. This API verifies that token and scopes
every query to the resulting user id -- the service key never leaves the server.

Run with:  uv run uvicorn app.server:app --reload --port 8000
Docs at:   http://localhost:8000/docs
"""
import io
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import (
    Body, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from app import db, inference

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",   # vite preview
]
MAX_BYTES = 50 * 1024 * 1024   # matches the 50MB cap advertised in the upload UI


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print("Loading the four trained hi-res models...")
    inference.load_models()
    print(f"Ready on {inference.H.DEVICE}. Serving {len(inference.ARCHS)} architectures.")

    st = db.status()
    if st["connected"]:
        print("  supabase: connected -- scans will persist")
        if not st.get("migrated"):
            print("  supabase: scan_predictions missing -- run "
                  "supabase/002_scan_predictions.sql")
    elif st["configured"]:
        print(f"  supabase: configured but unreachable -- {st.get('detail')}")
    else:
        print("  supabase: not configured -- results stay in the browser session")
    yield
    inference._MODELS.clear()


API_VERSION = "2.1.0"   # bumped when the request/response contract changes

app = FastAPI(title="AstraScan AI Inference API", version=API_VERSION,
              lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def current_user(authorization: Optional[str] = Header(None)) -> str:
    """Resolve the caller from their Supabase access token."""
    if not db.is_configured():
        raise HTTPException(status_code=503, detail="Storage is not configured.")
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    user_id = db.user_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not signed in.")
    return user_id


# ---------------------------------------------------------------------------
# meta -- unauthenticated
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": API_VERSION,
        "device": str(inference.H.DEVICE),
        "modelsLoaded": len(inference._MODELS),
        "storage": db.status(),
    }


@app.get("/api/models")
def models():
    """Metadata for the four architectures -- checkpoint name, decision threshold,
    and which one drives the headline verdict."""
    _, meta = inference.load_models(verbose=False)
    return {
        "primaryArch": inference.PRIMARY_ARCH,
        "device": str(inference.H.DEVICE),
        "models": [meta[a] for a in inference.ARCHS],
    }


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------
@app.get("/api/profile")
def get_profile(user_id: str = Depends(current_user)):
    profile = db.get_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found.")
    return profile


@app.patch("/api/profile")
def patch_profile(payload: dict = Body(...), user_id: str = Depends(current_user)):
    return db.update_profile(user_id, payload)


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
@app.get("/api/records")
def list_records(user_id: str = Depends(current_user)):
    return {"persisted": True, "records": db.list_records(user_id)}


@app.get("/api/records/{record_id}")
def get_record(record_id: str, user_id: str = Depends(current_user)):
    row = db.get_record(user_id, record_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Record not found.")
    return row


@app.patch("/api/records/{record_id}")
def patch_record(record_id: str, payload: dict = Body(...),
                 user_id: str = Depends(current_user)):
    if "notes" not in payload:
        raise HTTPException(status_code=400, detail="Only `notes` can be updated.")
    if not db.update_notes(user_id, record_id, payload["notes"] or ""):
        raise HTTPException(status_code=404, detail="Record not found.")
    return db.get_record(user_id, record_id)


@app.delete("/api/records/{record_id}")
def remove_record(record_id: str, user_id: str = Depends(current_user)):
    if not db.delete_record(user_id, record_id):
        raise HTTPException(status_code=404, detail="Record not found.")
    return {"deleted": record_id}


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------
@app.post("/api/analyze")
def analyze(file: UploadFile = File(...),
            patient_name: str = Form(""),
            user_id: str = Depends(current_user)):
    """Classify one mammogram with all four architectures, upload the Grad-CAM
    overlays to Storage, and persist the scan for the signed-in clinician."""
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 50MB limit.")

    try:
        pil_gray = Image.open(io.BytesIO(raw)).convert("L")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image. Send PNG, JPEG, BMP, or TIFF.",
        )

    try:
        analysis = inference.analyze(pil_gray, file_name=file.filename or "")
    except Exception as exc:  # surfaced to the UI rather than a bare 500
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    name = (file.filename or "scan").rsplit(".", 1)[0]
    try:
        record = db.save_analysis(user_id, name, analysis,
                                  patient_name=patient_name.strip() or None)
    except Exception as exc:
        # The one setup step that is easy to miss deserves a real instruction
        # rather than PostgREST's "schema cache" wording.
        if "scan_predictions" in str(exc):
            raise HTTPException(
                status_code=503,
                detail="Database is missing the scan_predictions table. Run "
                       "supabase/002_scan_predictions.sql in the Supabase SQL "
                       "Editor, then try again.",
            )
        raise HTTPException(status_code=500, detail=f"Saving the scan failed: {exc}")
    return record
