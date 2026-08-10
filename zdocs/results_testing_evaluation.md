# Chapter 4 (Excerpt): Results Presentation, Testing, and Evaluation

> Scope note: This document presents **Experiment 1 — Hi-Res Single-Input Architecture Comparison and Grad-CAM Interpretability Validation** only. All numbers, tables, and figures below are pulled directly from the experiment's saved result files (`notebooks/experiment_1_baseline/results/` and `notebooks/explainability/results/` and `figures/`), not re-derived or estimated. Sections for **Experiment 2 (Hyperparameter Optimization)** are left as placeholders for a later pass.

---

## 4.2 Results Presentation

### 4.2.1 System Output

The deployed system takes a single full mammogram image (any resolution) and produces two outputs: **(1)** a malignancy probability from the classifier, and **(2)** a Grad-CAM heatmap showing which regions of the image drove that prediction. No cropped lesion patch or segmentation mask is required at this stage — both are training-time-only signals (see [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md), §4.1).

Figures 4.1 and 4.2 show representative, unedited system output on five held-out test cases (three malignant, two benign) for the two architectures at opposite ends of the interpretability ranking established in §4.2.2: DenseNet121 (highest classification accuracy) and VGG-16 (highest localization quality).

**Figure 4.1.** Grad-CAM output for DenseNet121 on five test cases, columns left to right: original mammogram, mammogram with expert ROI outline overlaid, Grad-CAM heatmap alone, and heatmap + ROI combined.
![DenseNet121 Grad-CAM sample output](../notebooks/explainability/figures/batch_1/gradcam_overlay_grid_densenet121.png)

**Figure 4.2.** Same five test cases and layout as Figure 4.1, for VGG-16.
![VGG-16 Grad-CAM sample output](../notebooks/explainability/figures/batch_1/gradcam_overlay_grid_vgg16.png)

Reading these outputs qualitatively:

- **Row 1 (malignant, p=0.65 / p=0.40):** both architectures place heat near the annotated lesion, but DenseNet121 also lights up a large, unrelated region at the bottom of the breast — extra activation that is not wrong for the classification decision (GAP pooling only needs *some* discriminative signal) but would mislead a clinician using the heatmap as a location cue.
- **Row 3 (malignant, p=0.50 / p=0.54):** VGG-16's heatmap stays largely dark/cool over the annotated lesion — a localization miss despite a correct classification, illustrating why accuracy and interpretability must be measured separately rather than assumed to move together.
- **Row 5 (benign, correctly scored low at p=0.06):** both architectures peak on the laterality marker text ("L CC") burned into the corner of the image rather than on breast tissue. This is a genuine failure mode — the network is latching onto a non-anatomical artifact — and is exactly the kind of error that only becomes visible through explainability tooling, not through accuracy alone. It directly motivates the quantitative "peak-in-padding" and pixel-level localization checks reported in §4.2.2.

### 4.2.2 Data Visualization

**Classification performance.** All four architectures were trained under an identical recipe (384×640 input, batch size 8, seed 42, GAP pooling, distillation α = 0.5) and evaluated on the same held-out 319-image test split.

**Table 4.1.** Single-input (deployable) classification performance by architecture, with bootstrap 95% confidence intervals (2000 resamples).

| Architecture | Accuracy [95% CI] | AUC-ROC [95% CI] | Sensitivity | Specificity |
|---|---|---|---|---|
| **DenseNet121** | **0.7524** [0.705, 0.799] | **0.8390** [0.796, 0.879] | 0.7246 | 0.7735 |
| VGG-16 | 0.7273 [0.677, 0.774] | 0.8328 [0.789, 0.873] | 0.7391 | 0.7182 |
| EfficientNet-B2 | 0.7147 [0.664, 0.765] | 0.8133 [0.767, 0.858] | 0.6957 | 0.7293 |
| ResNet-50 | 0.7022 [0.652, 0.749] | 0.7939 [0.744, 0.840] | 0.6884 | 0.7127 |

**Figure 4.3.** Accuracy, AUC-ROC, sensitivity, and specificity by architecture (same data as Table 4.1).
![Classification metrics by architecture](../notebooks/experiment_1_baseline/results/hires_metrics_by_architecture_control_gap.png)

**Interpretability / localization performance.** Grad-CAM heatmaps were scored against the expert-annotated ROI masks on the same 319-image test set using the metrics defined in [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) §4.3: **energy concentration** (fraction of heatmap "mass" falling inside the lesion mask), **concentration ratio** (energy concentration ÷ chance rate — the primary, threshold-free metric), the **pointing game** (does the single hottest pixel land inside the mask?), and pixel-level **AP/AUROC** (treating the heatmap as a per-pixel malignancy score).

**Table 4.2.** Grad-CAM localization quality by architecture (n = 319 test images).

| Architecture | Concentration ratio (×chance) | Pointing game | Energy concentration | Pixel AP | Pixel AUROC | Peak-in-padding |
|---|---|---|---|---|---|---|
| **VGG-16** | **5.92×** [4.91, 7.05] | 16.93% | 4.74% | 0.152 | 0.661 | 2.51% |
| ResNet-50 | 4.26× [3.56, 4.97] | 9.09% | 3.66% | 0.090 | 0.655 | 1.25% |
| EfficientNet-B2 | 3.38× [2.86, 3.95] | 8.78% | 3.20% | 0.075 | 0.621 | 1.57% |
| DenseNet121 | 2.74× [2.28, 3.26] | 9.72% | 2.72% | 0.097 | 0.611 | 1.88% |

**Figure 4.4.** Left: distribution of per-image concentration ratio by architecture (box plot; dashed line = chance level, 1×). Right: IoU between the thresholded heatmap and the ROI mask as the threshold τ varies, on a 60-image illustrative subsample.
![Localization summary](../notebooks/explainability/figures/localization_summary.png)

**Figure 4.5.** Per-image concentration-ratio distribution split by ground-truth class (malignant vs. benign), by architecture.
![Localization by class per architecture](../notebooks/explainability/figures/localization_by_class_per_arch.png)

> **Note on Figure 4.4 (right panel):** the IoU-vs-threshold curve is computed on a 60-image subsample for visualization cost reasons and should be read as illustrative only. It shows DenseNet121 with a higher IoU across most thresholds than Table 4.2's full-sample (n=319) concentration-ratio ranking would suggest — a reminder that IoU is threshold-dependent and more sample-sensitive than the threshold-free concentration ratio, which is why concentration ratio (not IoU) is used as the primary, statistically-tested metric in this thesis (§4.2.3).

**Deployability comparison (single-input vs. dual-input).** Table 4.3 restates the core deployability result from [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) §5: how much accuracy is given up by dropping the ground-truth lesion crop at inference time.

**Table 4.3.** Cost of dropping the crop at inference, controlled comparison on the identical 319-row test split.

| Architecture | Single-input acc. | Dual-input acc. | Δ accuracy [95% CI] | McNemar p |
|---|---|---|---|---|
| DenseNet121 | 0.7524 | 0.7868 | −0.0345 [−0.088, +0.019] | 0.254 |
| VGG-16 | 0.7273 | 0.7524 | −0.0251 [−0.078, +0.034] | 0.445 |
| EfficientNet-B2 | 0.7147 | 0.7743 | −0.0596 [−0.122, +0.003] | 0.064 |
| ResNet-50 | 0.7022 | 0.7210 | −0.0188 [−0.075, +0.038] | 0.598 |

**Figure 4.6.** Left: test accuracy of the dual-input (non-deployable) model vs. the single-input model at the old 224² cache vs. the current 384×640 cache, against the majority-class floor. Center: accuracy gap lost by dropping the crop. Right: validation ROC-AUC training curves for all four single-input architectures.
![Hi-res vs dual-input comparison](../notebooks/experiment_1_baseline/results/hires_comparison_control_gap.png)

All four McNemar p-values are ≥ 0.05: **no architecture shows a statistically detectable accuracy loss from dropping the crop.** This null result is the central quantitative evidence for the thesis's deployability argument.

### 4.2.3 Performance Metrics

Table 4.4 summarizes the headline accuracy/AUC metrics from Table 4.1 alongside compute cost (wall-clock training time to convergence) as a proxy for computational demand — the resource dimension actually measured in this experiment.

**Table 4.4.** Classification performance and training compute cost by architecture.

| Architecture | Accuracy | AUC-ROC | Training time (min) | Best epoch (early-stopped) |
|---|---|---|---|---|
| DenseNet121 | 0.7524 | 0.8390 | 71.7 | 19 |
| VGG-16 | 0.7273 | 0.8328 | 37.3 | 10 |
| EfficientNet-B2 | 0.7147 | 0.8133 | 34.4 | 17 |
| ResNet-50 | 0.7022 | 0.7939 | 37.7 | 20 |

DenseNet121 achieves the best accuracy/AUC but at roughly **2× the training cost** of the other three architectures and the slowest convergence relative to its own final epoch count — a practical trade-off worth naming explicitly if DenseNet121 is recommended as the deployed model.

**Not yet measured:** per-image inference latency (response time) and memory footprint at inference time were not formally benchmarked in this experiment — only training-time GPU memory was a constraint (see §4.3.3). This is flagged here rather than estimated, and is a natural addition for a future pass of this section.

---

## 4.3 Testing and Evaluation

### 4.3.1 Testing Approach

Because this is a machine-learning system rather than a conventional application, "testing" is adapted from the standard unit/integration/system/acceptance hierarchy into an ML-appropriate equivalent: correctness checks on the *data* (leakage, alignment), correctness checks on the *model plumbing* (architecture shape assertions, sanity baselines), evaluation on a *held-out system-level test set*, and *statistical acceptance criteria* rather than pass/fail thresholds alone.

```mermaid
flowchart TD
    A["Data-integrity testing\n(patient-level leakage checks,\nimage/mask path resolution)"] --> B["Model unit validation\n(spatial-shape assertions,\nrandom-init Grad-CAM control)"]
    B --> C["System-level evaluation\n(held-out 319-image test set,\nnever used in training/tuning)"]
    C --> D["Statistical acceptance testing\n(McNemar for deployability,\nWilcoxon + Holm-Bonferroni\nfor interpretability ranking)"]
    D --> E{"Result"}
    E -->|"p >= 0.05"| F["No detectable cost -> deployable claim supported"]
    E -->|"p < 0.05 after correction"| G["Real, significant difference -> reported as a finding"]
```

**Figure 4.7.** Adapted testing pipeline used for Experiment 1.

### 4.3.2 Test Cases and Scenarios

**Table 4.5.** Representative test cases from Experiment 1, adapted to an ML evaluation context.

| Test ID | Description | Input | Expected Output | Actual Output | Status |
|---|---|---|---|---|---|
| TC01 | Patient-level leakage check | Train/test patient ID sets after `StratifiedGroupKFold` split | Zero patient ID overlap between splits | Zero overlap confirmed | Pass |
| TC02 | Backbone spatial-shape assertion | Each of the 4 backbones' pre-pool feature map at 384×640 input | Feature map shape consistent with `feat_dim` and expected spatial resolution | Assertion passed for all 4 architectures | Pass |
| TC03 | Grad-CAM sanity control | Untrained (random-weight) model, Grad-CAM run on a 40-image subsample | Concentration ratio ≈ chance (≈1×) — heatmap should carry no lesion signal before training | Near-chance concentration ratio observed, confirming Table 4.2's scores reflect learned features, not dataset geometry | Pass |
| TC04 | Deployability equivalence — DenseNet121 | Single-input vs. dual-input predictions, 319-image test set | No significant accuracy gap (McNemar p ≥ 0.05) | p = 0.254 | Pass |
| TC05 | Deployability equivalence — VGG-16 | same as TC04 | p ≥ 0.05 | p = 0.445 | Pass |
| TC06 | Deployability equivalence — EfficientNet-B2 | same as TC04 | p ≥ 0.05 | p = 0.064 | Pass |
| TC07 | Deployability equivalence — ResNet-50 | same as TC04 | p ≥ 0.05 | p = 0.598 | Pass |
| TC08 | Interpretability ranking significance | Paired concentration-ratio scores per architecture, n=319, all 6 pairwise comparisons | Wilcoxon signed-rank + Holm-Bonferroni correction determines which architecture pairs differ significantly | All 6 pairs significant (p_holm from 6.6×10⁻³ to 1.9×10⁻⁹); full ranking VGG-16 > ResNet-50 > EfficientNet-B2 > DenseNet121 | Pass (informative result — see discussion below) |
| TC09 | Heatmap-to-original-resolution round trip | Full-resolution test image of arbitrary aspect ratio, through `preprocess` → `GradCAM` → `cam_to_original` | Returned heatmap shape matches the *original* image's (H, W), not the internal 640×384 | Verified via `explain()` return-shape checks across the test set | Pass |

TC08 is the most consequential result in this table: it is a **statistically significant reversal** of the accuracy ranking in Table 4.1. DenseNet121, the most accurate architecture, is the *least* interpretable by every distance/overlap-based metric in Table 4.2; VGG-16, the second-most accurate, is the *most* interpretable. This is reported here as a genuine finding, not a test failure — it directly supports the thesis's premise that accuracy and explainability must be evaluated as separate axes rather than assumed to correlate.

### 4.3.3 Performance Evaluation

Two resource dimensions were exercised during Experiment 1:

- **Training compute time**, reported per architecture in Table 4.4 (34–72 minutes to convergence, DenseNet121 the most expensive).
- **GPU memory**, which directly constrained batch size: moving the input cache from 224² to 384×640 increases the pixel count per image by roughly 4.9×. At the chosen batch size of 8, this was projected to use approximately 4.8 GB of the available 5.76 GB of GPU memory — batch size 16 was therefore ruled out at this resolution, a scalability ceiling documented directly in the training configuration rather than discovered by trial and error.

No dedicated inference-latency or throughput benchmark (images/second at serving time) was run in this experiment; convergence speed (`best_epoch` in Table 4.4) is reported instead as the closest available proxy for "how much compute is needed to reach the reported result," and remains a gap to close in future work.

### 4.3.4 Comparison with Existing Systems

No external, published CBIS-DDSM benchmark comparison is included here — doing so honestly requires citing specific papers with matching train/test protocols, which is out of scope for this results pass and should be added separately with proper citations. What Experiment 1 *does* provide is an internal baseline comparison that is arguably more rigorous than a cross-paper comparison, because both systems were trained and evaluated on the exact same data split:

**Table 4.6.** Proposed (deployable) system vs. internal upper-bound baseline (dual-input, non-deployable), best architecture only.

| System | Inputs required at inference | Accuracy | AUC-ROC | Training time (min) |
|---|---|---|---|---|
| **Proposed system** (DenseNet121, single-input) | Full mammogram only | 0.7524 | 0.8390 | 71.7 |
| Internal baseline (DenseNet121, dual-input teacher) | Full mammogram + expert-annotated lesion crop | 0.7868 | 0.8655 | (not directly comparable — teacher trained separately, at 224²) |

As established in Table 4.3 and §4.2.2, the ~3.4-point accuracy gap between these two rows is **not statistically significant** (McNemar p = 0.254) — the practical reading of Table 4.6 is that the proposed system matches its own upper-bound baseline within measurement noise, while requiring strictly less information at inference time.

---

## Experiment 2 — Hyperparameter Optimization

*(To be completed. This experiment tunes the dual-input DenseNet121 configuration across six sequential phases — learning rate, weight decay, LR scheduler, augmentation strength, class-imbalance handling, and optimizer — followed by a test-time-augmentation pass; see `notebooks/experiment_2_optimization/hyperparameter_tuning.ipynb`. Results sections 4.2 and 4.3 for this experiment will be added once its result artifacts are finalized.)*
