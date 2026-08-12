"""Shared inference core for the AstraScan AI frontend.

Loads the four trained hi-res single-input models from experiment 1
(models/v4_hires_control_gap/, produced by
notebooks/experiment_1_baseline/hires_architecture_comparison.ipynb) and runs
classification + Grad-CAM. DenseNet121 is served from the sensitivity-optimized
checkpoint in notebooks/experiment_2_optimization/checkpoints/, which carries its
own `decision_threshold` (0.305) in the checkpoint metadata.

No training happens here -- this is the same inference path as app/app.py, just
factored out so the HTTP layer in app/server.py and the Streamlit demo can share
it.
"""
import base64
import io
import os
import sys
import threading
from pathlib import Path

import cv2
import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def _find_repo_root(markers=("models", "dataframes", "notebooks")):
    d = Path(__file__).resolve().parent
    for _ in range(6):
        if all((d / m).is_dir() for m in markers):
            return d
        d = d.parent
    raise RuntimeError("could not locate the repo root (expected models/, dataframes/, notebooks/)")


ROOT = _find_repo_root()
sys.path.insert(0, str(ROOT / "notebooks" / "common"))
import hires_lib as H  # noqa: E402

CKPT_DIR = ROOT / "models" / "v4_hires_control_gap"
EXP2_CKPT_DIR = ROOT / "notebooks" / "experiment_2_optimization" / "checkpoints"

ARCHS = H.ARCHITECTURES                       # densenet121, resnet50, efficientnet_b2, vgg16
PRIMARY_ARCH = "densenet121"                  # drives the headline verdict shown in the UI
DEFAULT_THRESHOLD = 0.50                      # every architecture except DenseNet121
DISPLAY_MAX_SIDE = 760                        # long-edge cap for the PNGs sent to the browser

# DenseNet121 is served from the high-sensitivity threshold variant saved by
# notebooks/experiment_2_optimization/sensitivity_threshold_adjustment.ipynb -- identical weights
# to the plain v4 checkpoint, but its metadata carries decision_threshold=0.305, which we read
# below and use as that model's cutoff. The other three keep the fixed 0.50.
CKPT_PATHS = {arch: CKPT_DIR / f"hires_control_gap_a0.5_{arch}.pth" for arch in ARCHS}
CKPT_PATHS[PRIMARY_ARCH] = (
    EXP2_CKPT_DIR / "hires_control_gap_a0.5_densenet121_high_sensitivity_t0.305.pth"
)

_MODELS = {}
_META = {}
_ROI_LOOKUP = None
_LOCK = threading.Lock()   # Grad-CAM registers hooks on the shared model; serialize requests


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_models(verbose=True):
    """Load all four checkpoints once. Returns (models, meta) keyed by arch."""
    if _MODELS:
        return _MODELS, _META

    for arch in ARCHS:
        ckpt_path = CKPT_PATHS[arch]
        if not ckpt_path.exists():
            raise FileNotFoundError(f"{arch}: checkpoint not found at {ckpt_path}")

        ck = torch.load(ckpt_path, map_location=H.DEVICE, weights_only=False)
        assert list(ck.get("input_size", [])) == [H.HIRES_H, H.HIRES_W], (
            f"{arch}: checkpoint input_size {ck.get('input_size')} != {[H.HIRES_H, H.HIRES_W]}")

        model = H.SingleInputModel(arch,
                                   pooling=ck.get("pooling", "gap"),
                                   use_seg=ck.get("use_seg", False)).to(H.DEVICE)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()

        # A checkpoint-embedded threshold wins over the 0.50 default -- that is how the
        # sensitivity-optimized DenseNet121 gets its 0.305 cutoff.
        threshold = ck.get("decision_threshold") or DEFAULT_THRESHOLD

        _MODELS[arch] = model
        _META[arch] = {
            "arch": arch,
            "displayName": H.ARCH_DISPLAY_NAMES[arch],
            "checkpoint": ckpt_path.name,
            "checkpointDir": str(ckpt_path.parent.relative_to(ROOT)).replace("\\", "/"),
            "threshold": float(threshold),
            "thresholdVariant": ck.get("threshold_variant"),
            "valAuc": ck.get("val_auc"),
            "testMetrics": ck.get("threshold_test_metrics"),
            "isPrimary": arch == PRIMARY_ARCH,
        }
        if verbose:
            variant = _META[arch]["thresholdVariant"]
            tag = f" [{variant}]" if variant else ""
            print(f"  loaded {H.ARCH_DISPLAY_NAMES[arch]:<16} t={threshold:.3f}{tag}  {ckpt_path.name}")

    return _MODELS, _META


def build_roi_lookup():
    """basename(source PNG) -> {roi_hires_path, pathology}, across the train/val/test hi-res
    dataframes. Ground-truth ROIs exist only for CBIS-DDSM images; an externally supplied
    mammogram has no annotation to look up."""
    global _ROI_LOOKUP
    if _ROI_LOOKUP is not None:
        return _ROI_LOOKUP

    lookup = {}
    for split in ("train", "val", "test"):
        csv_path = os.path.join(H.DF_DIR, f"{split}_df_hires.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            key = os.path.basename(str(row["image file path"])).lower()
            lookup[key] = {"roi_hires_path": row[H.ROI_HIRES_COL], "pathology": row[H.LABEL_COL]}
    _ROI_LOOKUP = lookup
    return _ROI_LOOKUP


# ---------------------------------------------------------------------------
# image helpers -- identical geometry to the training cache
# ---------------------------------------------------------------------------
def preprocess(pil_gray):
    """Raw PIL image -> (1,3,640,384) normalized tensor, matching the training cache exactly."""
    canvas = H.fit_pad(pil_gray, Image.LANCZOS)
    t = H.TF.to_tensor(canvas)
    return H.TF.normalize(t.expand(3, -1, -1).clone(), H.IMAGENET_MEAN, H.IMAGENET_STD).unsqueeze(0)


def downscale_for_display(arr, mode, max_side=DISPLAY_MAX_SIDE):
    h, w = arr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return arr
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    return np.asarray(Image.fromarray(arr, mode=mode).resize((nw, nh), Image.BILINEAR))


def make_overlay(gray_arr_u8, heat_arr, alpha=0.45):
    """Grayscale image + jet heatmap, alpha-blended. Both arrays already at display size."""
    base = np.stack([gray_arr_u8] * 3, axis=-1).astype(np.float32) / 255.0
    h = heat_arr.astype(np.float32)
    peak = h.max()
    if peak > 0:
        h = h / peak
    colored = mpl.colormaps["jet"](h)[..., :3]
    blended = np.clip((1 - alpha) * base + alpha * colored, 0, 1)
    return Image.fromarray((blended * 255).astype(np.uint8))


def roi_mask_at_original(ow, oh, roi_hires_path):
    """The cached ROI mask lives in padded 640x384 space. Strip the padding with the same geometry
    fit_pad used to build it, then resize back to the uploaded image's own resolution."""
    nw, nh, ox, oy = H.fit_pad_geometry(ow, oh)
    padded = np.asarray(Image.open(roi_hires_path).convert("L")) > 127
    core = (padded[oy:oy + nh, ox:ox + nw].astype(np.uint8)) * 255
    return np.asarray(Image.fromarray(core, mode="L").resize((ow, oh), Image.NEAREST)) > 127


def draw_roi_overlay(gray_arr_u8, mask_bool, color=(57, 219, 92)):
    """Grayscale display image + a green contour around the ground-truth ROI."""
    rgb = np.stack([gray_arr_u8] * 3, axis=-1).copy()
    mask_u8 = mask_bool.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thickness = max(1, gray_arr_u8.shape[0] // 300)
    cv2.drawContours(rgb, contours, -1, color, thickness=thickness)
    return Image.fromarray(rgb)


def to_data_uri(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------
def explain(model, pil_gray):
    """Returns (prob_malignant, heatmap_at_original_resolution)."""
    ow, oh = pil_gray.size
    x = preprocess(pil_gray).to(H.DEVICE)
    with H.GradCAM(model) as cam_fn:
        cam, probs = cam_fn(x, class_idx=1)
    return float(probs[0, 1]), H.cam_to_original(cam[0], ow, oh)


def analyze(pil_gray, file_name=""):
    """Run all four architectures on one mammogram.

    Returns a JSON-serializable dict: the headline verdict from the
    sensitivity-optimized DenseNet121, a per-architecture breakdown with Grad-CAM
    overlays, and the ground-truth ROI when the filename matches a CBIS-DDSM row.
    """
    models, meta = load_models(verbose=False)

    with _LOCK:
        orig_display = downscale_for_display(np.asarray(pil_gray), mode="L")

        results = []
        for arch in ARCHS:
            prob, heat = explain(models[arch], pil_gray)
            heat_disp = downscale_for_display(heat, mode="F")
            overlay = make_overlay(orig_display, heat_disp)

            threshold = meta[arch]["threshold"]
            malignant = prob >= threshold
            results.append({
                **meta[arch],
                "probability": round(prob, 4),
                "result": "Malignant" if malignant else "Benign",
                # confidence in the prediction the model actually made
                "confidence": round((prob if malignant else 1 - prob) * 100, 1),
                "heatmap": to_data_uri(overlay),
            })

        # ground truth, when this file is a known CBIS-DDSM image
        ground_truth = None
        match = build_roi_lookup().get(os.path.basename(file_name).lower()) if file_name else None
        if match:
            ow, oh = pil_gray.size
            mask_full = roi_mask_at_original(ow, oh, match["roi_hires_path"])
            mask_disp = downscale_for_display(mask_full.astype(np.uint8) * 255, mode="L") > 127
            ground_truth = {
                "available": True,
                "label": "Malignant" if H.encode_label(match["pathology"]) == 1 else "Benign",
                "roiImage": to_data_uri(draw_roi_overlay(orig_display, mask_disp)),
            }

    primary = next(r for r in results if r["arch"] == PRIMARY_ARCH)
    return {
        "fileName": file_name,
        "device": str(H.DEVICE),
        "originalImage": to_data_uri(Image.fromarray(orig_display, mode="L")),
        "groundTruth": ground_truth,
        "primaryArch": PRIMARY_ARCH,
        "result": primary["result"],
        "probability": primary["probability"],
        "confidence": primary["confidence"],
        "threshold": primary["threshold"],
        "models": results,
    }
