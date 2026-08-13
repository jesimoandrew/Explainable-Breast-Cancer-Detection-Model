"""Shared library for the hi-res single-input mammography experiments.

Single source of truth for models, data, metrics, and Grad-CAM. Previously `compute_metrics` and
the model definitions were copy-pasted across four notebooks, which guarantees drift.

HARD CONSTRAINT this code exists to serve: the deployed model takes ONLY the full mammogram at
inference. The cropped lesion patch and the ROI mask are training-time signals only -- the crop
via a frozen teacher's soft labels, the mask via an auxiliary segmentation decoder. Neither is
required to produce a prediction or a Grad-CAM heatmap.
"""
from __future__ import annotations

import gc
import os
import random
import sys
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

Image.MAX_IMAGE_PIXELS = None

# Paths, derived from this file's location so notebooks work from any subdirectory.
_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_LIB_DIR, "..", ".."))
DF_DIR = os.path.join(PROJECT_ROOT, "dataframes")
TEACHER_CKPT_DIR = os.path.join(PROJECT_ROOT, "notebooks", "experiment_1_baseline", "checkpoints")
LOGIT_DIR = os.path.join(DF_DIR, "teacher_logits")

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
SEED = 42
HIRES_W, HIRES_H = 384, 640          # must match build_hires_cache.py TARGET_W/TARGET_H
LEGACY_SIZE = 224                    # what the frozen dual-input teachers were trained at

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# hi-res columns (from build_hires_cache.py); legacy 224 columns (for the teacher)
FULL_HIRES_COL = "full hires cached path"
ROI_HIRES_COL = "roi hires cached path"
FULL_224_COL = "full image cached path"
CROP_224_COL = "cropped image cached path"
LABEL_COL = "pathology"
CLASS_NAMES = ["Benign", "Malignant"]

ARCHITECTURES = ["densenet121", "resnet50", "efficientnet_b2", "vgg16"]
ARCH_DISPLAY_NAMES = {
    "densenet121": "DenseNet121",
    "resnet50": "ResNet-50",
    "efficientnet_b2": "EfficientNet-B2",
    "vgg16": "VGG-16",
}
FEAT_DIMS = {"densenet121": 1024, "resnet50": 2048, "efficientnet_b2": 1408, "vgg16": 512}

# weight_decay=1e-4 matches the recorded baselines. experiment_2's sweep selected 1e-5 but it TIED
# on val loss (0.451265 either way), so continuity with the existing numbers wins. Whatever is
# chosen must be identical across every arm.
BASELINE = dict(head_lr=1e-4, backbone_lr=1e-5, weight_decay=1e-4, scheduler="plateau")

BATCH_SIZE = 8               # 384x640 is 4.9x the pixels of 224^2; bs16 projects to ~4.8GB of 5.76GB
NUM_WORKERS = 4 if os.name != "nt" else 0
PIN_MEMORY = False           # ~210MB of page-locked host RAM; host RAM is the scarce resource here
HEAD_WARMUP_EPOCHS = 5
MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 8

DISTILL_TEMPERATURE = 4.0
DISTILL_ALPHA = 0.5
SEG_LOSS_WEIGHT = 0.3
SEG_DICE_EPS = 1.0
# Measured on the hi-res cache: mask foreground median 0.42%, mean 0.87%. sigmoid(-5.5) ~= 0.0041.
SEG_BIAS_INIT = -5.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def encode_label(raw_label) -> int:
    try:
        return int(raw_label)
    except (ValueError, TypeError):
        pass
    return 1 if "MALIGNANT" in str(raw_label).strip().upper() else 0


# --------------------------------------------------------------------------------------
# Preprocessing geometry -- and its inverse
# --------------------------------------------------------------------------------------
def fit_pad_geometry(orig_w: int, orig_h: int, tw: int = HIRES_W, th: int = HIRES_H):
    """The exact geometry build_hires_cache.fit_pad applied. Deterministic in the source size,
    which is what makes the Grad-CAM inverse possible."""
    s = min(tw / orig_w, th / orig_h)
    nw, nh = max(1, round(orig_w * s)), max(1, round(orig_h * s))
    return nw, nh, (tw - nw) // 2, (th - nh) // 2


def fit_pad(im: Image.Image, resample, tw: int = HIRES_W, th: int = HIRES_H) -> Image.Image:
    """Scale to fit inside tw x th preserving aspect, centre on a black canvas. Identical to
    build_hires_cache.fit_pad -- duplicated here so inference can preprocess a raw image without
    importing the cache builder."""
    nw, nh, ox, oy = fit_pad_geometry(im.size[0], im.size[1], tw, th)
    canvas = Image.new("L", (tw, th), 0)
    canvas.paste(im.resize((nw, nh), resample), (ox, oy))
    return canvas


def validity_mask(orig_w: int, orig_h: int, tw: int = HIRES_W, th: int = HIRES_H) -> np.ndarray:
    """Boolean (th, tw) marking the pasted region, i.e. everything that is NOT padding.

    Needed because 7-12% of columns (up to 54%) in this cache are pure padding. A Grad-CAM argmax
    landing in padding is meaningless, so localization metrics must be restricted to this region."""
    nw, nh, ox, oy = fit_pad_geometry(orig_w, orig_h, tw, th)
    v = np.zeros((th, tw), dtype=bool)
    v[oy:oy + nh, ox:ox + nw] = True
    return v


def cam_to_original(cam: np.ndarray, orig_w: int, orig_h: int,
                    tw: int = HIRES_W, th: int = HIRES_H) -> np.ndarray:
    """Map a (th, tw) heatmap back to the ORIGINAL image resolution.

    The production requirement: the explanation must be overlayable on the image the user supplied,
    not on the 384x640 preprocessed tensor. Strips the padding the preprocessing added, then
    resizes the remaining core back to (orig_w, orig_h).

    Memory: a 3024x5063 float32 result is ~61MB. Call this one image at a time. Quantitative
    metrics should be computed in 384x640 space instead, where the cached mask already lives.
    """
    nw, nh, ox, oy = fit_pad_geometry(orig_w, orig_h, tw, th)
    core = np.asarray(cam, dtype=np.float32)[oy:oy + nh, ox:ox + nw]
    return np.asarray(Image.fromarray(core, mode="F").resize((orig_w, orig_h), Image.BILINEAR),
                      dtype=np.float32)


# --------------------------------------------------------------------------------------
# Backbones
# --------------------------------------------------------------------------------------
def _assert_spatial(extractor: nn.Module, arch: str, dim: int,
                    hw=(HIRES_H, HIRES_W)) -> tuple:
    """Assert the extractor emits a genuine spatial map -- catches slicing mistakes like resnet50's
    [:-1] (which leaves avgpool in place and silently yields 1x1).

    Runs under eval() and restores the prior mode: a probe forward in train() mode would update the
    backbone's pretrained BatchNorm running stats from a zeros input, corrupting the very weights
    about to be fine-tuned.
    """
    was_training = extractor.training
    extractor.eval()
    try:
        with torch.no_grad():
            shape = tuple(extractor(torch.zeros(1, 3, *hw)).shape)
    finally:
        extractor.train(was_training)
    assert len(shape) == 4 and shape[1] == dim and shape[-1] > 1 and shape[-2] > 1, \
        f"{arch}: expected spatial (1, {dim}, >1, >1) at {hw}, got {shape}"
    return shape


def build_backbone_spatial(arch: str, verify_hw=(HIRES_H, HIRES_W)):
    """(extractor, feat_dim) where extractor(x) is the PRE-POOL spatial map. This is the tensor
    Grad-CAM reads and the seg decoder consumes, so it must not be pooled away."""
    if arch == "densenet121":
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        extractor, dim = nn.Sequential(base.features, nn.ReLU(inplace=True)), 1024
    elif arch == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # [:-2] not [:-1]: resnet's own avgpool sits before fc, so [:-1] still returns (B,2048,1,1)
        extractor, dim = nn.Sequential(*list(base.children())[:-2]), 2048
    elif arch == "efficientnet_b2":
        base = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        extractor, dim = base.features, 1408
    elif arch == "vgg16":
        base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        extractor, dim = base.features, 512
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    if verify_hw is not None:
        _assert_spatial(extractor, arch, dim, verify_hw)
    return extractor, dim


def build_backbone_pooled(arch: str):
    """Pooled variant -- must match architecture_comparison.ipynb exactly so the frozen dual-input
    teacher checkpoints (stage0_baseline_{arch}.pth) load."""
    if arch == "densenet121":
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        return nn.Sequential(base.features, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1))), 1024
    if arch == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        return nn.Sequential(*list(base.children())[:-1]), 2048
    if arch == "efficientnet_b2":
        base = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        return nn.Sequential(base.features, nn.AdaptiveAvgPool2d((1, 1))), 1408
    if arch == "vgg16":
        base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        return nn.Sequential(base.features, nn.AdaptiveAvgPool2d((1, 1))), 512
    raise ValueError(f"Unknown architecture: {arch}")


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
class DualInputModel(nn.Module):
    """The frozen teacher. Takes full + crop, so it is NOT deployable -- the crop is cut from the
    ground-truth ROI annotation. Used only to generate soft labels during training."""

    def __init__(self, arch, num_classes=2, dropout=0.5):
        super().__init__()
        self.arch = arch
        self.full_extractor, d1 = build_backbone_pooled(arch)
        self.crop_extractor, d2 = build_backbone_pooled(arch)
        self.classifier = nn.Sequential(
            nn.Linear(d1 + d2, 512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, full_image, cropped_image):
        f = torch.flatten(self.full_extractor(full_image), 1)
        c = torch.flatten(self.crop_extractor(cropped_image), 1)
        return self.classifier(torch.cat([f, c], dim=1))


def load_teacher(arch, ckpt_dir):
    teacher = DualInputModel(arch).to(DEVICE)
    ckpt = torch.load(os.path.join(ckpt_dir, f"stage0_baseline_{arch}.pth"),
                      map_location=DEVICE, weights_only=False)
    teacher.load_state_dict(ckpt["model_state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


class SingleInputModel(nn.Module):
    """The deployable student: full mammogram in, class logits out.

    pooling:
      'gap'      global average pool (the 224 baseline's behaviour)
      'gap_gmp'  concat of average and max pool. At 384x640 the map is 12x20 = 240 cells with the
                 lesion in ~1 of them, so GAP dilutes the lesion signal ~200x AND averages over a
                 per-image-varying amount of black padding. GMP supplies a padding-immune,
                 lesion-sensitive path. Strictly contains GAP as a subspace, so it cannot lose
                 capacity, and adds no new hyperparameters.

    use_seg adds a training-only decoder predicting the ROI mask. It is never needed to classify;
    forward() ignores it entirely.
    """

    def __init__(self, arch, num_classes=2, dropout=0.5, pooling="gap", use_seg=False,
                 verify_hw=(HIRES_H, HIRES_W)):
        super().__init__()
        assert pooling in ("gap", "gap_gmp"), pooling
        self.arch, self.pooling, self.use_seg = arch, pooling, use_seg
        self.extractor, feat_dim = build_backbone_spatial(arch, verify_hw)
        self.feat_dim = feat_dim
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        head_in = feat_dim * (2 if pooling == "gap_gmp" else 1)
        self.classifier = nn.Sequential(
            nn.Linear(head_in, 512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        if use_seg:
            self.seg_decoder = nn.Sequential(
                nn.Conv2d(feat_dim, 256, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
            )
            # Prior-informed bias: the target is >99% background, so a naive 50/50 init produces
            # huge unstable early gradients (cf. RetinaNet focal-loss init).
            nn.init.constant_(self.seg_decoder[-1].bias, SEG_BIAS_INIT)

    def _pool(self, spatial):
        a = torch.flatten(self.avgpool(spatial), 1)
        if self.pooling == "gap":
            return a
        return torch.cat([a, torch.flatten(self.maxpool(spatial), 1)], dim=1)

    def forward(self, full_image):
        """Classification only. THIS is the inference path."""
        return self.classifier(self._pool(self.extractor(full_image)))

    def forward_with_seg(self, full_image, seg_out_hw=None):
        """Training-only. Segmentation logits are supervised at reduced resolution -- the decoder's
        information content is capped by the 12x20 feature grid, so upsampling to full resolution
        before the loss is pure compute with no added signal."""
        spatial = self.extractor(full_image)
        logits = self.classifier(self._pool(spatial))
        if not self.use_seg:
            return logits, None
        hw = seg_out_hw or (full_image.shape[-2] // 4, full_image.shape[-1] // 4)
        seg = F.interpolate(self.seg_decoder(spatial), size=hw, mode="bilinear", align_corners=False)
        return logits, seg

    def set_backbone_trainable(self, trainable: bool):
        for p in self.extractor.parameters():
            p.requires_grad = trainable
        self.extractor.train(trainable)

    def head_parameters(self):
        """Everything that is NOT the backbone. Phase 2 builds an explicit param-group list, which
        does not auto-include new modules the way Phase 1's filter(requires_grad) does -- the
        seg_decoder was silently frozen at epoch 6 by exactly this bug once already."""
        mods = [self.classifier] + ([self.seg_decoder] if self.use_seg else [])
        return [p for m in mods for p in m.parameters()]


def assert_optimizer_complete(optimizer, model):
    """Every trainable parameter must be in some param group. Catches the 'new module silently
    stops training when Phase 2 begins' class of bug outright."""
    in_opt = {id(p) for g in optimizer.param_groups for p in g["params"]}
    want = {id(p) for p in model.parameters() if p.requires_grad}
    missing = want - in_opt
    assert not missing, f"{len(missing)} trainable params missing from the optimizer"


# --------------------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------------------
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    hard = F.cross_entropy(student_logits, labels)
    if alpha <= 0.0:
        return hard
    soft_t = F.softmax(teacher_logits / temperature, dim=1)
    soft_s = F.log_softmax(student_logits / temperature, dim=1)
    soft = F.kl_div(soft_s, soft_t, reduction="batchmean") * (temperature ** 2)
    return alpha * soft + (1 - alpha) * hard


def dice_loss_from_logits(logits, target, eps=SEG_DICE_EPS):
    probs = torch.sigmoid(logits).flatten(1)
    target = target.flatten(1)
    inter = (probs * target).sum(1)
    return 1 - ((2 * inter + eps) / (probs.sum(1) + target.sum(1) + eps)).mean()


def seg_loss_fn(seg_logits, mask_targets):
    """BCE + Dice. At ~0.4% foreground, BCE alone lets the decoder collapse to 'all background' at
    near-zero loss; Dice penalizes poor overlap regardless of imbalance.
    binary_cross_entropy_with_logits is the autocast-safe form (plain BCE is not)."""
    return (F.binary_cross_entropy_with_logits(seg_logits, mask_targets)
            + dice_loss_from_logits(seg_logits, mask_targets))


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
_color_jitter = transforms.ColorJitter(brightness=0.15, contrast=0.15)


def paired_geometric_augment(full_img, mask_img):
    """One random draw shared by image and mask. The segmentation loss is a pixel-wise comparison,
    so independent draws would corrupt the target outright. TF.rotate defaults to NEAREST, matching
    RandomRotation(15)'s default and keeping the mask strictly binary."""
    if random.random() < 0.5:
        full_img, mask_img = TF.hflip(full_img), TF.hflip(mask_img)
    angle = transforms.RandomRotation.get_params([-15, 15])
    return TF.rotate(full_img, angle), TF.rotate(mask_img, angle)


def _to_rgb_tensor(pil_l):
    """Load as L then expand to 3 channels on the TENSOR -- .convert('RGB') on the PIL image
    allocates 3x the bytes during decode, which matters on a 7.2GB machine."""
    t = TF.to_tensor(pil_l)                      # (1,H,W)
    t = TF.normalize(t.expand(3, -1, -1).clone(), IMAGENET_MEAN, IMAGENET_STD)
    return t


class HiResDataset(Dataset):
    """Returns (full_hires, mask, teacher_logits, label).

    Teacher logits are PRECOMPUTED (offline distillation), so the 224 teacher never runs in the
    training loop: two image decodes per sample instead of four, and no teacher on the GPU.
    """

    def __init__(self, df, train: bool, teacher_logits: np.ndarray | None = None,
                 seg_out_hw=(HIRES_H // 4, HIRES_W // 4)):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.teacher_logits = teacher_logits
        self.seg_out_hw = seg_out_hw
        if teacher_logits is not None:
            assert len(teacher_logits) == len(self.df), \
                f"teacher logits {len(teacher_logits)} != rows {len(self.df)}"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        full = Image.open(row[FULL_HIRES_COL]).convert("L")
        mask = Image.open(row[ROI_HIRES_COL]).convert("L")

        if self.train:
            full, mask = paired_geometric_augment(full, mask)
            full = _color_jitter(full)

        full_t = _to_rgb_tensor(full)
        mask_t = (TF.to_tensor(mask) > 0.5).float()
        if self.seg_out_hw is not None:
            mask_t = F.interpolate(mask_t[None], size=self.seg_out_hw, mode="nearest")[0]

        tl = (torch.from_numpy(self.teacher_logits[idx]).float()
              if self.teacher_logits is not None else torch.zeros(2))
        return full_t, mask_t, tl, torch.tensor(encode_label(row[LABEL_COL]), dtype=torch.long)


def make_loaders(train_df, val_df, test_df, teacher_logits=None, batch_size=BATCH_SIZE):
    tl = teacher_logits or {}
    mk = lambda df, tr, key: DataLoader(
        HiResDataset(df, train=tr, teacher_logits=tl.get(key)), batch_size=batch_size,
        shuffle=tr, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=tr)
    return mk(train_df, True, "train"), mk(val_df, False, "val"), mk(test_df, False, "test")


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def compute_metrics(labels, preds, probs):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "sensitivity": recall_score(labels, preds, pos_label=1, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": auc,
    }


def metrics_at_threshold(labels, probs, threshold):
    return compute_metrics(labels, (np.asarray(probs) >= threshold).astype(int), probs)


def bootstrap_ci(labels, probs, metric="accuracy", threshold=0.5, n_boot=2000, seed=SEED):
    """95% CI. At n=319 the SE on accuracy is ~2.6pp, so a bare point estimate invites
    noise-chasing; '71.2% [66.1-76.3]' is defensible in a way '71.2%' is not."""
    rng = np.random.default_rng(seed)
    labels, probs = np.asarray(labels), np.asarray(probs)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[i])) < 2:
            continue
        vals.append(metrics_at_threshold(labels[i], probs[i], threshold)[metric])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --------------------------------------------------------------------------------------
# Grad-CAM
# --------------------------------------------------------------------------------------
class GradCAM:
    """Grad-CAM on SingleInputModel.extractor, via hooks so the model itself is untouched.

    Using hooks (rather than adding a forward_features method) means the identical code path runs
    on control and seg-aux checkpoints, so the localization comparison cannot be confounded by a
    code difference. It also works on DualInputModel.full_extractor, which has the same interface.

    Two requirements that silently produce garbage if violated:
      * AMP must be OFF -- fp16 gradients underflow and yield all-zero CAMs that look like model
        failure rather than a numerical one.
      * eval() mode but WITH grad enabled -- the usual no_grad evaluation context is wrong here.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.layer = target_layer if target_layer is not None else model.extractor
        self._act = self._grad = self._handle = None

    def __enter__(self):
        def fwd_hook(_m, _i, out):
            self._act = out
            # Tensor-level hook on the Sequential's OUTPUT. A module backward hook would trip over
            # the inplace ReLU inside densenet's extractor.
            if out.requires_grad:
                out.register_hook(lambda g: setattr(self, "_grad", g))
        self._handle = self.layer.register_forward_hook(fwd_hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()          # a leaked hook retains activations -> host-RAM leak
        self._act = self._grad = self._handle = None
        return False

    def __call__(self, x, class_idx=1, out_hw=(HIRES_H, HIRES_W)):
        """Returns (cam[B,H,W] in [0,1], probs[B,2]).

        class_idx defaults to 1 (malignant) and should stay fixed rather than following argmax:
        with argmax, benign and malignant cases explain different targets, making the localization
        metric incomparable across classes. It also matches what a deployed tool shows.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.enable_grad():
                # The INPUT must carry grad. Grad-CAM needs gradients w.r.t. the activations, not
                # w.r.t. the weights -- and a frozen model (load_teacher sets every param to
                # requires_grad=False) builds no graph at all unless the input does. Without this,
                # the extractor output has requires_grad=False and cannot even be hooked.
                x = x.detach().clone().requires_grad_(True)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).detach()
                self.model.zero_grad(set_to_none=True)
                logits[:, class_idx].sum().backward()

                assert self._act is not None, \
                    "no activations captured -- is the target layer on the forward path?"
                assert self._grad is not None, \
                    "no gradients captured -- the activation tensor had requires_grad=False"
                w = self._grad.mean(dim=(2, 3), keepdim=True)
                cam = F.relu((w * self._act).sum(dim=1, keepdim=True))
                # NOTE size=(H, W). The cache is H=640, W=384; transposing these silently "works".
                cam = F.interpolate(cam, size=out_hw, mode="bilinear", align_corners=False)[:, 0]
                # Divide by a CLAMPED max, not (max + eps). With (max + eps) the normalized peak is
                # m/(m+eps), which only approaches 1.0 for large raw activations -- so peak height
                # would vary with activation magnitude and CAM values would not be comparable
                # across images. That breaks any fixed-tau thresholding (e.g. the IoU metric).
                # clamp_min gives an exact 1.0 peak whenever the max is positive, and leaves a
                # genuinely all-zero CAM as all-zero rather than dividing by eps.
                cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
                return cam.detach().cpu().numpy(), probs.cpu().numpy()
        finally:
            self.model.train(was_training)
            self.model.zero_grad(set_to_none=True)


def localization_metrics(cam, mask, valid, tolerance_px=15):
    """How well does a Grad-CAM heatmap agree with the ground-truth ROI? All arrays (H, W).

    Returns pointing-game hits, energy concentration (threshold-free, the headline), and the
    concentration ratio -- energy concentration divided by the chance rate, i.e. "the CAM puts N
    times more of its attention on the lesion than random would".
    """
    cam = np.asarray(cam, dtype=np.float64)
    mask = np.asarray(mask).astype(bool)
    valid = np.asarray(valid).astype(bool)
    out = {}

    raw_peak = np.unravel_index(np.argmax(cam), cam.shape)
    out["peak_in_padding"] = not bool(valid[raw_peak])

    masked = np.where(valid, cam, -np.inf)
    py, px = np.unravel_index(np.argmax(masked), masked.shape)
    out["pointing_hit"] = bool(mask[py, px])

    if mask.any():
        ys, xs = np.nonzero(mask)
        out["peak_dist_px"] = float(np.min(np.hypot(ys - py, xs - px)))
        out["pointing_hit_tol"] = bool(out["peak_dist_px"] <= tolerance_px)
    else:
        out["peak_dist_px"] = float("nan")
        out["pointing_hit_tol"] = False

    denom = cam[valid].sum()
    out["energy_concentration"] = float(cam[mask & valid].sum() / denom) if denom > 0 else float("nan")
    chance = mask[valid].sum() / max(valid.sum(), 1)
    out["chance_rate"] = float(chance)
    out["concentration_ratio"] = (out["energy_concentration"] / chance) if chance > 0 else float("nan")

    if mask[valid].any() and not mask[valid].all():
        out["pixel_ap"] = float(average_precision_score(mask[valid], cam[valid]))
        out["pixel_auroc"] = float(roc_auc_score(mask[valid], cam[valid]))
    else:
        out["pixel_ap"] = out["pixel_auroc"] = float("nan")
    return out


def iou_at(cam, mask, valid, tau):
    b = (np.asarray(cam) >= tau) & valid
    m = np.asarray(mask).astype(bool) & valid
    union = (b | m).sum()
    return float((b & m).sum() / union) if union else float("nan")


def dice_at(cam, mask, valid, tau):
    """Sorensen-Dice of the tau-thresholded CAM against the ROI.

    Dice and IoU are the SAME ordering per image -- Dice = 2*IoU/(1+IoU) exactly, because
    |A|+|B| = |A union B| + |A intersect B|. Both are reported because both are conventional in the
    segmentation literature, not because they are independent evidence. Their MEANS over a dataset
    do differ (the transform is nonlinear), so the two columns are not redundant as summaries, but
    a per-image win for one is always a win for the other.
    """
    b = (np.asarray(cam) >= tau) & valid
    m = np.asarray(mask).astype(bool) & valid
    tot = b.sum() + m.sum()
    return float(2 * (b & m).sum() / tot) if tot else float("nan")


def overlap_sweep(cam, mask, valid, taus):
    """IoU and Dice at every tau in `taus`, in a single pass. Returns {"tau", "iou", "dice"} arrays.

    Sorting the valid CAM values once and locating each tau with searchsorted is far cheaper than
    re-thresholding a 640x384 array per tau -- the grid is swept for every image, every
    architecture and both splits, so the naive version dominates the runtime.

    Equivalent to calling iou_at/dice_at in a loop; `test_overlap_sweep_matches_naive` in the
    notebook asserts that equivalence rather than trusting it.
    """
    valid = np.asarray(valid).astype(bool)
    v = np.asarray(cam, dtype=np.float64)[valid]
    m = np.asarray(mask).astype(bool)[valid]
    order = np.argsort(v, kind="stable")
    vs = v[order]
    # cum_m[i] = lesion pixels among the i COLDEST values, so the lesion pixels at or above a
    # threshold sitting at index i is simply M - cum_m[i].
    cum_m = np.concatenate([[0], np.cumsum(m[order])])
    M, n = int(m.sum()), int(vs.size)

    taus = np.asarray(taus, dtype=np.float64)
    idx = np.searchsorted(vs, taus, side="left")
    b_n = n - idx                                  # |predicted positive|
    inter = M - cum_m[idx]                         # |predicted positive AND lesion|
    union = b_n + M - inter
    tot = b_n + M
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
        dice = np.where(tot > 0, 2 * inter / np.maximum(tot, 1), np.nan)
    return {"tau": taus, "iou": iou.astype(np.float64), "dice": dice.astype(np.float64)}


# --------------------------------------------------------------------------------------
# Memory probe
# --------------------------------------------------------------------------------------
def probe_memory(arch, batch_size=BATCH_SIZE, hw=(HIRES_H, HIRES_W), use_seg=True,
                 pooling="gap_gmp", steps=3):
    """Run real training steps and report peak VRAM. Two minutes here prevents an eight-hour sweep
    dying at hour six."""
    if not torch.cuda.is_available():
        return {"arch": arch, "peak_gb": float("nan"), "note": "no cuda"}
    free_gpu()
    torch.cuda.reset_peak_memory_stats()
    model = opt = None
    try:
        model = SingleInputModel(arch, pooling=pooling, use_seg=use_seg, verify_hw=hw).to(DEVICE)
        model.set_backbone_trainable(True)
        opt = torch.optim.AdamW([
            {"params": model.extractor.parameters(), "lr": 1e-5},
            {"params": model.head_parameters(), "lr": 1e-4},
        ], weight_decay=1e-4)
        assert_optimizer_complete(opt, model)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        x = torch.randn(batch_size, 3, *hw, device=DEVICE)
        m = (torch.rand(batch_size, 1, hw[0] // 4, hw[1] // 4, device=DEVICE) > 0.99).float()
        y = torch.randint(0, 2, (batch_size,), device=DEVICE)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                logits, seg = model.forward_with_seg(x)
                loss = F.cross_entropy(logits, y)
                if seg is not None:
                    loss = loss + SEG_LOSS_WEIGHT * seg_loss_fn(seg, m)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        return {"arch": arch, "batch_size": batch_size, "peak_gb": round(peak, 2), "ok": peak < 4.5}
    except BaseException:
        traceback.clear_frames(sys.exc_info()[2])
        raise
    finally:
        del model, opt
        free_gpu()


# --------------------------------------------------------------------------------------
# Teacher logits (offline distillation)
# --------------------------------------------------------------------------------------
# Recorded dual-input teacher performance on the *_final test split (326 rows). The hi-res split
# has 319 rows, so this is a sanity band rather than an equality check. Wildly off (~0.5) means the
# checkpoint or preprocessing path is broken.
TEACHER_REFERENCE = {"densenet121": (0.7791, 0.8617)}


class TeacherPairDataset(Dataset):
    """The 224 full+crop pair the teacher was trained on. No augmentation: a soft label should be a
    fixed property of a sample, not a function of a random draw."""

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        out = []
        for col in (FULL_224_COL, CROP_224_COL):
            t = TF.to_tensor(Image.open(row[col]).convert("L"))
            out.append(TF.normalize(t.expand(3, -1, -1).clone(), IMAGENET_MEAN, IMAGENET_STD))
        return out[0], out[1], torch.tensor(encode_label(row[LABEL_COL]), dtype=torch.long)


def compute_teacher_logits(arch, dfs, batch_size=16, check_reference=True):
    """Run the frozen 224 teacher once over every split and cache the logits.

    This removes the teacher from the training loop entirely: two image decodes per sample instead
    of four, and no second model resident on the GPU.

    `check_reference` gates on the teacher reproducing its recorded score. Disable it only for
    truncated splits, where head(n) is not a random sample and the comparison is meaningless.
    """
    os.makedirs(LOGIT_DIR, exist_ok=True)
    path = os.path.join(LOGIT_DIR, f"teacher_logits_{arch}.npz")
    if os.path.exists(path):
        z = np.load(path)
        if all(k in z for k in dfs) and all(len(z[k]) == len(dfs[k]) for k in dfs):
            print(f"[teacher] using cached logits: {os.path.basename(path)}")
            return {k: z[k] for k in dfs}
        print("[teacher] cached logits stale (row count mismatch) -- recomputing")

    teacher, out = None, {}
    try:
        teacher = load_teacher(arch, TEACHER_CKPT_DIR)
        for split, df in dfs.items():
            dl = DataLoader(TeacherPairDataset(df), batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
            logits, labels = [], []
            with torch.no_grad():
                for f, c, y in tqdm(dl, desc=f"teacher {split}", leave=False):
                    logits.append(teacher(f.to(DEVICE), c.to(DEVICE)).cpu().numpy())
                    labels.append(y.numpy())
            out[split] = np.concatenate(logits).astype(np.float32)
            lab = np.concatenate(labels)
            probs = torch.softmax(torch.from_numpy(out[split]), 1)[:, 1].numpy()
            m = compute_metrics(lab, out[split].argmax(1), probs)
            print(f"[teacher] {split:5s} n={len(lab):5d} acc {m['accuracy']:.4f} auc {m['roc_auc']:.4f}")
            if split == "test" and arch in TEACHER_REFERENCE:
                ra, ru = TEACHER_REFERENCE[arch]
                print(f"[teacher] reference (326-row _final split): acc {ra:.4f} auc {ru:.4f}")
                if not check_reference:
                    print("[teacher] reference check SKIPPED (truncated split)")
                elif abs(m["accuracy"] - ra) >= 0.05 or abs(m["roc_auc"] - ru) >= 0.05:
                    raise AssertionError(
                        f"teacher scored acc {m['accuracy']:.4f} / auc {m['roc_auc']:.4f}, far from "
                        f"the recorded {ra}/{ru}. The checkpoint or preprocessing path is broken -- "
                        "every distilled run downstream would be poisoned.")
                else:
                    print("[teacher] reference check PASSED")
    finally:
        del teacher
        free_gpu()

    if check_reference:
        np.savez_compressed(path, **out)
        print(f"[teacher] wrote {path}")
    else:
        print("[teacher] truncated split -- logits NOT cached")
    return out


# --------------------------------------------------------------------------------------
# Epoch loops
# --------------------------------------------------------------------------------------
def run_epoch(model, loader, device, temperature, alpha, seg_weight,
              optimizer=None, scaler=None, desc=""):
    """One pass. Training when `optimizer` is given, evaluation otherwise.

    Also reports how often the student is right where the teacher is wrong, and the reverse. Once
    student_beats_teacher consistently exceeds teacher_beats_student, the student has overtaken its
    teacher and the KL term is pulling it toward the teacher's errors -- the signal to try alpha=0.
    """
    train = optimizer is not None
    model.train(train)
    tot = {"loss": 0.0, "cls": 0.0, "seg": 0.0}
    preds, labs, probs, tpred = [], [], [], []
    n = 0
    with (torch.enable_grad() if train else torch.no_grad()):
        for full, mask, tlog, y in tqdm(loader, desc=desc, leave=False):
            full, mask = full.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            tlog, y = tlog.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, seg = model.forward_with_seg(full)
                cls = distillation_loss(logits, tlog, y, temperature, alpha)
                sl = seg_loss_fn(seg, mask) if seg is not None else torch.zeros((), device=device)
                loss = cls + (seg_weight * sl if seg is not None else 0.0)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            bs = y.size(0)
            n += bs
            tot["loss"] += loss.item() * bs
            tot["cls"] += cls.item() * bs
            tot["seg"] += sl.detach().item() * bs      # detach: sl carries grad while training
            preds.extend(logits.argmax(1).detach().cpu().numpy())
            labs.extend(y.detach().cpu().numpy())
            probs.extend(torch.softmax(logits.float(), 1)[:, 1].detach().cpu().numpy())
            tpred.extend(tlog.argmax(1).detach().cpu().numpy())

    m = compute_metrics(labs, preds, probs)
    labs, preds, tpred = np.array(labs), np.array(preds), np.array(tpred)
    m["student_beats_teacher"] = float(((preds == labs) & (tpred != labs)).mean())
    m["teacher_beats_student"] = float(((preds != labs) & (tpred == labs)).mean())
    return {k: v / max(n, 1) for k, v in tot.items()}, m


@torch.no_grad()
def seg_dice(model, loader, device, thresh=0.5):
    """Mean Dice of the auxiliary decoder. At 224 this reached only ~0.065 -- because the lesion
    was a quarter of one 7x7 cell and literally not representable. At 20x12 it is representable, so
    this number is the real test of whether mask supervision is doing anything."""
    if not getattr(model, "use_seg", False):
        return float("nan")
    model.eval()
    tot, n = 0.0, 0
    for full, mask, _, _ in tqdm(loader, desc="seg dice", leave=False):
        full, mask = full.to(device), mask.to(device)
        _, seg = model.forward_with_seg(full)
        p = (torch.sigmoid(seg) > thresh).float().flatten(1)
        t = mask.flatten(1)
        d = (2 * (p * t).sum(1) + 1e-6) / (p.sum(1) + t.sum(1) + 1e-6)
        tot += d.sum().item()
        n += len(d)
    return tot / max(n, 1)
