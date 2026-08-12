"""Streamlit demo: upload a full mammogram, get a malignancy classification and a
side-by-side Grad-CAM comparison across all four trained architectures.

No training happens here. This loads the already-trained v4 checkpoints from
models/v4_hires_control_gap/ and runs inference + Grad-CAM only -- the same
production path documented in notebooks/explainability/gradcam_validation.ipynb.

Run with: streamlit run app/app.py
"""
import os
import sys
from pathlib import Path

import cv2
import matplotlib as mpl
import numpy as np
import pandas as pd
import streamlit as st
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
ARCHS = H.ARCHITECTURES
DISPLAY_MAX_SIDE = 800  # long-edge cap for on-screen overlays; the model still sees the image at full resolution via fit_pad

# Per-architecture checkpoint override. DenseNet121 uses the "high_sensitivity" threshold-variant
# checkpoint saved in notebooks/experiment_2_optimization/sensitivity_threshold_adjustment.ipynb --
# identical weights to the plain v4 checkpoint, but it carries a `decision_threshold` in its
# metadata (see load_models below), which this app reads and uses as that model's default cutoff.
CKPT_PATHS = {arch: CKPT_DIR / f"hires_control_gap_a0.5_{arch}.pth" for arch in ARCHS}
CKPT_PATHS["densenet121"] = EXP2_CKPT_DIR / "hires_control_gap_a0.5_densenet121_high_sensitivity_t0.305.pth"

DEFAULT_THRESHOLD = 0.50  # fixed cutoff for every architecture except DenseNet121

# From notebooks/experiment_2_optimization/sensitivity_threshold_adjustment.ipynb -- tuned for
# DenseNet121 specifically, offered here as presets since that analysis exists.
PRESETS = {
    "Main (t=0.50)": 0.50,
    "Crossover (t=0.485)": 0.485,
    "High-Sensitivity (t=0.305)": 0.305,
}

st.set_page_config(page_title="Mammogram Malignancy Classifier", layout="wide")


@st.cache_data(show_spinner=False)
def build_roi_lookup():
    """basename(source PNG) -> {roi_hires_path, pathology}, for every row across the train/val/test
    hi-res dataframes. Ground-truth ROI masks only exist for images already in the CBIS-DDSM
    dataset -- an arbitrary externally-supplied mammogram has no annotation to look up."""
    lookup = {}
    for split in ("train", "val", "test"):
        csv_path = os.path.join(H.DF_DIR, f"{split}_df_hires.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            key = os.path.basename(str(row["image file path"])).lower()
            lookup[key] = {"roi_hires_path": row[H.ROI_HIRES_COL], "pathology": row[H.LABEL_COL]}
    return lookup


def roi_mask_at_original(ow, oh, roi_hires_path):
    """The cached ROI mask lives in padded 640x384 space. Strip the padding using the same
    geometry `fit_pad` used to build it, then resize the remaining core back to the uploaded
    image's own resolution -- mirrors H.cam_to_original but for a binary mask."""
    nw, nh, ox, oy = H.fit_pad_geometry(ow, oh)
    padded = np.asarray(Image.open(roi_hires_path).convert("L")) > 127
    core = (padded[oy:oy + nh, ox:ox + nw].astype(np.uint8)) * 255
    return np.asarray(Image.fromarray(core, mode="L").resize((ow, oh), Image.NEAREST)) > 127


def draw_roi_overlay(gray_arr_u8, mask_bool, color=(57, 219, 92)):
    """Grayscale display image + a green contour around the ground-truth ROI (display resolution)."""
    rgb = np.stack([gray_arr_u8] * 3, axis=-1).copy()
    mask_u8 = mask_bool.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thickness = max(1, gray_arr_u8.shape[0] // 300)
    cv2.drawContours(rgb, contours, -1, color, thickness=thickness)
    return Image.fromarray(rgb)


@st.cache_resource(show_spinner="Loading the four trained models...")
def load_models():
    loaded, meta = {}, {}
    for arch in ARCHS:
        ckpt_path = CKPT_PATHS[arch]
        ck = torch.load(ckpt_path, map_location=H.DEVICE, weights_only=False)
        assert list(ck.get("input_size", [])) == [H.HIRES_H, H.HIRES_W], (
            f"{arch}: checkpoint input_size {ck.get('input_size')} != {[H.HIRES_H, H.HIRES_W]}")
        model = H.SingleInputModel(arch, pooling=ck.get("pooling", "gap"),
                                    use_seg=ck.get("use_seg", False)).to(H.DEVICE)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        loaded[arch] = model
        meta[arch] = {"checkpoint_name": ckpt_path.name,
                      "threshold_variant": ck.get("threshold_variant"),
                      "decision_threshold": ck.get("decision_threshold")}
    return loaded, meta


def preprocess(pil_gray):
    """Raw PIL image -> (1,3,640,384) normalized tensor, identical geometry to the training cache,
    so a model trained on the cache sees exactly this at inference."""
    canvas = H.fit_pad(pil_gray, Image.LANCZOS)
    t = H.TF.to_tensor(canvas)
    return H.TF.normalize(t.expand(3, -1, -1).clone(), H.IMAGENET_MEAN, H.IMAGENET_STD).unsqueeze(0)


def explain(model, pil_gray):
    """Returns (prob_malignant, heatmap_at_original_resolution)."""
    ow, oh = pil_gray.size
    x = preprocess(pil_gray).to(H.DEVICE)
    with H.GradCAM(model) as cam_fn:
        cam, probs = cam_fn(x, class_idx=1)
    return float(probs[0, 1]), H.cam_to_original(cam[0], ow, oh)


def downscale_for_display(arr, mode, max_side=DISPLAY_MAX_SIDE):
    h, w = arr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return arr
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    img = Image.fromarray(arr, mode=mode).resize((nw, nh), Image.BILINEAR)
    return np.asarray(img)


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


st.title("Mammogram Malignancy Classifier")
st.caption(
    "Upload a full mammogram. Shows the original image, its ground-truth ROI annotation (when "
    "available), and the classification + Grad-CAM heatmap from each of the four deployable "
    "single-input models (DenseNet121, ResNet-50, EfficientNet-B2, VGG-16) -- six images in "
    "total. Only the full mammogram is used at inference -- no lesion crop or ROI mask -- "
    "matching the deployed inference path; the ROI panel is ground truth shown for comparison "
    "only, never fed to a model."
)

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Mammogram image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

    st.header("Decision threshold -- DenseNet121 only")
    preset_label = st.selectbox("Preset", list(PRESETS) + ["Custom"], index=0)
    if preset_label == "Custom":
        densenet_threshold = st.slider("Malignant if p(malignant) >=", 0.0, 1.0, 0.50, 0.005)
    else:
        densenet_threshold = PRESETS[preset_label]
        st.slider("Malignant if p(malignant) >=", 0.0, 1.0, densenet_threshold, 0.005, disabled=True)
    st.caption(
        "Only affects the DenseNet121 panel -- these presets were tuned for DenseNet121 "
        "specifically (see notebooks/experiment_2_optimization/sensitivity_threshold_adjustment.ipynb). "
        f"ResNet-50, EfficientNet-B2, and VGG-16 stay fixed at the default {DEFAULT_THRESHOLD:.2f}."
    )
    st.caption(f"Device: {H.DEVICE}")

if uploaded is None:
    st.info("Upload a mammogram image to run the four models.")
    st.stop()

pil_gray = Image.open(uploaded).convert("L")
models, model_meta = load_models()

orig_display = downscale_for_display(np.asarray(pil_gray), mode="L")

roi_lookup = build_roi_lookup()
roi_match = roi_lookup.get(os.path.basename(uploaded.name).lower())

st.subheader("Original & Ground-Truth ROI")
top_cols = st.columns(2)
with top_cols[0]:
    st.image(orig_display, use_container_width=True, caption="Original")
with top_cols[1]:
    if roi_match:
        ow, oh = pil_gray.size
        mask_full = roi_mask_at_original(ow, oh, roi_match["roi_hires_path"])
        mask_disp = downscale_for_display(mask_full.astype(np.uint8) * 255, mode="L") > 127
        roi_overlay_img = draw_roi_overlay(orig_display, mask_disp)
        st.image(roi_overlay_img, use_container_width=True, caption="Ground-truth ROI (CBIS-DDSM annotation)")
        truth = "Malignant" if H.encode_label(roi_match["pathology"]) == 1 else "Benign"
        st.caption(f"Dataset ground-truth label: **{truth}**")
    else:
        st.info(
            "No ground-truth ROI available -- this filename wasn't matched to a CBIS-DDSM "
            "train/val/test row, so there's no annotation to overlay."
        )

st.subheader("Classification and Grad-CAM by architecture")
cols = st.columns(len(ARCHS))

for col, arch in zip(cols, ARCHS):
    with col:
        with st.spinner(f"{H.ARCH_DISPLAY_NAMES[arch]}..."):
            prob, heat = explain(models[arch], pil_gray)
        heat_disp = downscale_for_display(heat, mode="F")
        overlay = make_overlay(orig_display, heat_disp)

        arch_threshold = densenet_threshold if arch == "densenet121" else DEFAULT_THRESHOLD

        st.image(overlay, use_container_width=True, caption=H.ARCH_DISPLAY_NAMES[arch])
        if prob >= arch_threshold:
            st.error(f"**Malignant**  \np(malignant) = {prob:.3f}  (>= {arch_threshold:.3f})")
        else:
            st.success(f"**Benign**  \np(malignant) = {prob:.3f}  (< {arch_threshold:.3f})")

        variant = model_meta[arch]["threshold_variant"]
        if variant:
            embedded = model_meta[arch]["decision_threshold"]
            st.caption(
                f"Checkpoint: `{model_meta[arch]['checkpoint_name']}` -- "
                f"variant **{variant}**, embedded threshold {embedded:.3f} "
                f"(classified at the sidebar threshold above, adjustable for this model only)."
            )
