"""Re-cache CBIS-DDSM full mammograms + ROI masks at higher, aspect-preserving resolution.

Why: at the current square 224x224 cache the median lesion measures 19x10 px -- smaller than a
single cell of DenseNet121's 7x7 output grid. That caps both classification accuracy and Grad-CAM,
whose heatmap resolution IS the feature-map resolution.

Two changes here:
  1. Resolution 224 -> 640 on the long side.
  2. Aspect ratio PRESERVED (pad to canvas) instead of squashed into a square. Mammograms are
     ~0.6 w/h, so the square resize was compressing every lesion vertically by ~1.7x. Padding is
     black, which is what mammogram background already is, so it introduces no artifact.
Median lesion goes 19x10 -> 35x34 px (6.1x the area).

The CROP is deliberately NOT re-cached: it feeds the frozen dual-input teacher, which was trained
at 224. Feeding that teacher anything else is an untested distribution shift, and its soft labels
are the whole point of distillation. The student gets hi-res; the teacher keeps its 224 inputs.

Resumable: re-running skips files that already exist.
"""
import os
import sys
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = "cbis_ddsm_png"
OUT_ROOT = "cbis-ddsm-cache-hires"
DF_DIR = "dataframes"
TARGET_W, TARGET_H = 384, 640          # 0.60 aspect, matches the median mammogram
FULL_COL_OUT = "full hires cached path"
ROI_COL_OUT = "roi hires cached path"

_BS = chr(92)


def winlong(path):
    """Windows extended-length path -- CBIS-DDSM's nested DICOM UID folders blow past MAX_PATH."""
    path = os.path.abspath(path)
    return (_BS * 2 + "?" + _BS + path) if not path.startswith(_BS * 2) else path


def resolve_roi_mask(crop_rel_path):
    """The ROI mask is the sibling PNG sitting next to the crop in its annotation series folder,
    at full source resolution. Verified: the crop folder holds exactly two files -- the small crop
    patch and the full-size mask."""
    crop_rel_path = str(crop_rel_path).replace("/", os.sep)
    folder = os.path.join(ROOT, os.path.dirname(crop_rel_path))
    crop_name = os.path.basename(crop_rel_path)
    try:
        names = os.listdir(winlong(folder))
    except OSError:
        return None
    siblings = [n for n in names if n.lower().endswith(".png") and n != crop_name]
    if len(siblings) != 1:
        return None
    return os.path.join(folder, siblings[0])


def fit_pad(im, resample):
    """Scale to fit inside TARGET_W x TARGET_H preserving aspect, then centre on a black canvas.
    Returns the canvas. The scale factor and paste offset depend only on the input size, so calling
    this on an image and its mask (same source dims) keeps them pixel-aligned."""
    w, h = im.size
    s = min(TARGET_W / w, TARGET_H / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    canvas = Image.new("L", (TARGET_W, TARGET_H), 0)
    canvas.paste(im.resize((nw, nh), resample), ((TARGET_W - nw) // 2, (TARGET_H - nh) // 2))
    return canvas


def process_split(split):
    src_csv = os.path.join(DF_DIR, f"{split}_df_with_roi.csv")
    df = pd.read_csv(src_csv)
    full_dir = os.path.join(OUT_ROOT, split, "full")
    roi_dir = os.path.join(OUT_ROOT, split, "roi")
    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(roi_dir, exist_ok=True)

    full_paths, roi_paths, failures = [], [], []
    n = len(df)
    for i, row in df.iterrows():
        fo = os.path.abspath(os.path.join(full_dir, f"full_{i}.png"))
        ro = os.path.abspath(os.path.join(roi_dir, f"roi_{i}.png"))

        if os.path.exists(fo) and os.path.exists(ro):
            full_paths.append(fo); roi_paths.append(ro)
        else:
            try:
                fsrc = os.path.join(ROOT, str(row["image file path"]).replace("/", os.sep))
                msrc = resolve_roi_mask(row["cropped image file path"])
                if msrc is None:
                    raise FileNotFoundError("could not resolve ROI mask sibling")

                with Image.open(winlong(fsrc)) as im:
                    fim, fsize = im.convert("L"), im.size
                    fit_pad(fim, Image.LANCZOS).save(fo, optimize=True)
                with Image.open(winlong(msrc)) as im:
                    if im.size != fsize:
                        raise ValueError(f"mask {im.size} != full {fsize}; would misalign")
                    # NEAREST keeps the mask strictly binary -- no interpolated grey edges.
                    fit_pad(im.convert("L"), Image.NEAREST).save(ro, optimize=True)

                full_paths.append(fo); roi_paths.append(ro)
            except Exception as e:
                failures.append((i, repr(e)[:120]))
                full_paths.append(None); roi_paths.append(None)

        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"  {split}: {i + 1}/{n}  ({len(failures)} failed)", flush=True)

    df[FULL_COL_OUT] = full_paths
    df[ROI_COL_OUT] = roi_paths
    kept = df.dropna(subset=[FULL_COL_OUT, ROI_COL_OUT]).reset_index(drop=True)
    out_csv = os.path.join(DF_DIR, f"{split}_df_hires.csv")
    kept.to_csv(out_csv, index=False)
    print(f"  {split}: wrote {out_csv} with {len(kept)}/{n} rows", flush=True)
    for i, e in failures[:5]:
        print(f"     failure row {i}: {e}", flush=True)
    return len(kept), n, failures


if __name__ == "__main__":
    print(f"caching to {TARGET_W}x{TARGET_H} (aspect-preserved, black-padded)", flush=True)
    total = {}
    for split in ("train", "val", "test"):
        total[split] = process_split(split)
    print("\n=== summary ===")
    for split, (kept, n, fails) in total.items():
        print(f"  {split:5s} {kept}/{n} rows cached ({len(fails)} failures)")
    sizes = []
    for split in ("train", "val", "test"):
        for sub in ("full", "roi"):
            d = os.path.join(OUT_ROOT, split, sub)
            if os.path.isdir(d):
                sizes.append(sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)))
    print(f"  cache size on disk: {sum(sizes) / 1024 ** 2:.0f} MB")
