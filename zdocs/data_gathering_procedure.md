# Data Gathering Procedure

This study is entirely quantitative and is conducted in three sequential experiments on a single dataset, CBIS-DDSM. Each experiment consumes the artifacts produced by the one before it: Experiment 1 selects a baseline architecture and caches its test-set predictions, Experiment 2 re-reads those cached predictions to select a decision threshold, and Experiment 3 scores the resulting model's Grad-CAM explanations against expert-drawn ROI masks. No human raters, expert panels, or survey instruments are involved at any stage; all interpretability evidence is gathered *functionally* — by comparing machine-generated heatmaps against the ground-truth annotations already distributed with the dataset.

A shared preprocessing stage (§0) precedes all three experiments, since every experiment draws from the same cached images and the same patient-level splits.

---

## 0. Shared Data Preparation

**0.1 Dataset acquisition.** The CBIS-DDSM dataset (Curated Breast Imaging Subset of the Digital Database for Screening Mammography) is downloaded from its public repository. It is distributed as DICOM images accompanied by two metadata CSVs — `calc_case_description_*.csv` for calcifications and `mass_case_description_*.csv` for masses — in which each row describes one abnormality and carries a `patient_id`, a `pathology` label, and relative paths to the full mammogram, an expert-cropped lesion patch, and an ROI segmentation mask. All DICOM files are converted once to PNG for compatibility with the deep learning framework.

**0.2 Label binarization.** The `pathology` field is mapped to a binary target, with the CBIS-DDSM category `BENIGN_WITHOUT_CALLBACK` folded into `BENIGN`, yielding two classes: benign (0) and malignant (1).

**0.3 ROI mask resolution.** CBIS-DDSM stores the lesion crop and its ROI mask as two identically-named PNGs in the same folder, with no reliable naming convention distinguishing them. The mask is resolved by elimination: after removing the known crop file, exactly one PNG should remain. Rows whose mask cannot be unambiguously resolved are dropped rather than guessed at, since a mismatched mask would silently corrupt every localization metric computed in Experiment 3.

**0.4 Patient-level dataset splitting.** The cleaned dataset is divided into training (80%), validation (10%), and test (10%) subsets using scikit-learn's `StratifiedGroupKFold` with a fixed `random_state = 42`. This splitter groups by `patient_id` and stratifies by `pathology` simultaneously, so that (a) no patient's images ever cross a split boundary and (b) the benign/malignant proportion is preserved in each subset. Grouping is essential rather than optional here: a single patient contributes multiple images (both breasts, multiple views), so a naive row-level split would leak a patient's other images into the test set and inflate accuracy. CBIS-DDSM's own published train/test CSVs were additionally checked for patient overlap and found to leak, which is why the splits are rebuilt from scratch rather than adopted as-is. The resulting held-out test set contains 319 images and is used, unchanged, in all three experiments.

**0.5 Image preprocessing and caching.** Decoding DICOM on every training step would make training I/O-bound, so all images are preprocessed once into a fixed-size PNG cache. Each image is converted to grayscale and letterboxed onto a **384 × 640** canvas: scaled to fit while preserving aspect ratio, then centered on a black background. The full mammogram is resampled with Lanczos interpolation; the ROI mask is resampled with nearest-neighbor so that a binary mask stays binary instead of acquiring interpolated gray edges. Both pass through the *same* padding geometry, keeping mammogram and mask pixel-aligned — an alignment that Experiment 3 depends on. At load time, pixel intensities are scaled to [0, 1], expanded to three channels, and normalized with ImageNet statistics to match the pretrained backbones.

The 384 × 640 canvas (0.60 aspect ratio) was chosen to match the median mammogram's proportions. An earlier 224 × 224 square cache was rejected because at that resolution the median CBIS-DDSM lesion measures roughly 19 × 10 pixels — smaller than a single cell of the final 7 × 7 convolutional feature grid — which caps both classification accuracy and Grad-CAM resolution. At 384 × 640 the median lesion grows to approximately 35 × 34 pixels, about six times the pixel area, without distorting the image. No DICOM windowing, CLAHE, or denoising filter is applied; preprocessing is deliberately restricted to geometric resizing so that results reflect what a standard CNN backbone achieves on raw pixel intensities.

**0.6 Data augmentation.** Augmentation is applied to the training set only; validation and test sets are never augmented. Geometric augmentations — horizontal flip (p = 0.5) and random rotation within ±15° — are applied to the image and its ROI mask from a *single shared random draw*, so the two remain aligned. Photometric augmentation (brightness and contrast jitter, ±0.15) is applied to the image only, having no meaning for a binary mask.

---

## 1. Experiment 1 — Baseline Architecture Comparison

**Objective.** Determine which pretrained CNN backbone yields the best classification performance under an identical training recipe, and confirm that the deployable single-input formulation loses no detectable accuracy relative to a model that also sees the expert lesion crop.

**1.1 Two-stage teacher/student setup.** At deployment a clinician has only the full mammogram — no hand-cropped patch and no segmentation mask. Models are therefore trained in two stages. A **dual-input teacher** sees the full mammogram *and* the ground-truth lesion crop, giving it more information than will ever be available at inference; it is trained first, then frozen. A **single-input student** sees only the full mammogram and is trained with knowledge distillation from the teacher, blending cross-entropy against the true label with a KL-divergence term against the teacher's softened output distribution (temperature = 4.0, mixing weight α = 0.5, following Hinton et al., 2015). The student is the model that is actually deployed and explained; the teacher functions as an internal upper bound.

**1.2 Baseline model training.** Four ImageNet-pretrained architectures — **DenseNet121, ResNet-50, EfficientNet-B2, and VGG-16** — are each trained as single-input students under an identical configuration. Every backbone is truncated before its original classification head so that it still emits a spatial feature map (required both by the global-average-pooling classifier head and by Grad-CAM in Experiment 3). Training proceeds in two phases: a 5-epoch head warm-up with the backbone frozen (learning rate 1e-4), followed by full fine-tuning of up to 35 further epochs with a discriminative learning rate (backbone 1e-5, head 1e-4), `ReduceLROnPlateau` scheduling, and early stopping on validation loss with a patience of 8 epochs. Batch size 8, weight decay 1e-4, and random seed 42 are held constant. Holding input resolution, pooling, augmentation, optimizer, schedule, and seed identical across all four runs is what makes any observed performance difference attributable to the architecture rather than to the training regimen.

**1.3 Baseline model comparison.** Each trained model is evaluated once on the held-out 319-image test set using Accuracy, Sensitivity, Specificity, F1-Score, and AUC-ROC. Percentile bootstrap 95% confidence intervals (2,000 resamples) accompany the headline metrics, since a point estimate at n = 319 is not defensible on its own. Wall-clock training time and best epoch are recorded alongside the metrics as the compute-cost dimension of the comparison.

**1.4 Deployability check.** For each architecture, the single-input student's accuracy is compared against its dual-input teacher's on the identical test split, with the difference expressed as a **paired bootstrap 95% confidence interval** (2,000 resamples) on Δ accuracy. This quantifies how much accuracy is surrendered by dropping the expert crop at inference, and establishes whether the deployable formulation is distinguishable from its own upper bound at this sample size. An interval spanning zero is reported as *no detectable difference*, which is explicitly weaker than a claim of equivalence.

**1.5 Artifacts carried forward.** The best-performing architecture is designated the baseline for Experiment 2. Its trained checkpoint and its **cached test-set probability file** (`testprobs_*.npz`) are saved; Experiment 2 consumes the latter directly, and Experiment 3 the former.

---

## 2. Experiment 2 — Sensitivity Optimization

**Objective.** At the default 0.500 decision cutoff, the baseline model's sensitivity trails its specificity — that is, it is more likely to miss a malignancy than to raise a false alarm, which is the wrong direction for a cancer-screening deployment where a false negative is the costlier error. This experiment asks whether the *decision threshold*, rather than the model, can be adjusted to favor sensitivity.

**2.1 Threshold sweep.** Using the baseline model's cached test-set probabilities from Experiment 1, the classification threshold is swept across **198 fixed steps from 0.010 to 0.995**, recomputing Accuracy, Sensitivity, Specificity, and F1-Score at each cutoff. No retraining and no new inference are performed — every point on the sweep re-reads the same probability vector under a different cutoff. AUC-ROC is invariant by construction, since it summarizes ranking ability across all possible thresholds.

**2.2 Candidate threshold selection.** Two candidate operating points are read directly off the sweep with no manual tuning:

- **Crossover** — the largest threshold at which sensitivity still exceeds specificity, i.e. the smallest possible departure from the default.
- **High-Sensitivity** — the threshold maximizing accuracy among all thresholds satisfying sensitivity > specificity.

**2.3 Threshold evaluation.** Both candidates are evaluated against the default 0.500 threshold on the same 319-image test set, with percentile bootstrap 95% confidence intervals (2,000 resamples per candidate per metric). A candidate whose accuracy interval overlaps the default's is treated as a statistically cost-free shift toward sensitivity. Where a candidate is selected by argmax over the same test set on which it is reported, the resulting test-set selection bias is disclosed explicitly and the point estimate is presented as an illustrative upper bound rather than a validated final choice.

**2.4 Checkpoint finalization.** Each selected threshold is written out as a standalone checkpoint carrying the *identical* trained weights as the Experiment 1 source checkpoint plus a `decision_threshold` field and the corresponding test-set operating metrics as metadata. Weight integrity is verified tensor-by-tensor against the source checkpoint (`torch.equal` over every key in the `state_dict`), confirming that only the inference-time cutoff differs. The finalized checkpoint and its Grad-CAM module proceed to Experiment 3.

---

## 3. Experiment 3 — Explainability Evaluation

**Objective.** Measure — rather than eyeball — whether the model's Grad-CAM explanations point at the actual lesion, and determine which architecture explains itself best. Because CBIS-DDSM ships expert-drawn ROI masks, this evaluation is **functionally grounded**: the ground truth is the radiologist annotation already in the dataset, so no additional expert elicitation is required.

**3.1 Ground-truth mask preparation.** For every abnormality in the test set, the ROI segmentation mask is loaded from the 384 × 640 cache built in §0.5, binarized at an intensity of 127, and used directly — no rescaling is needed, since the mask was cached through the same padding geometry as its mammogram and is therefore already pixel-aligned with it. Images whose resolved mask is empty are skipped. A **validity mask** marking the non-padding region of the canvas is computed per image from the original source dimensions; 7–12% of the cache is black letterbox padding, and a heatmap peak landing there is meaningless.

**3.2 Heatmap generation.** Grad-CAM (Selvaraju et al., 2017) is implemented from scratch using forward and backward hooks on each backbone's final spatial feature map — the same tensor the classifier pools over, so the explanation is computed on exactly the features the prediction used. Each channel is weighted by its mean gradient of the malignant logit, the weighted sum is passed through ReLU to retain only class-supporting evidence, and the result is bilinearly upsampled to the 384 × 640 input resolution and normalized so that each map's peak is exactly 1.0. The target class is **pinned to malignant (class 1)** for every image rather than following the model's own top prediction, so that benign and malignant cases are both explained with respect to the same clinical question ("does anything here look malignant") instead of a benign case producing a "what looks benign" map. Heatmaps are generated for the same 319 test images used in Experiments 1–2, with the Experiment 2 checkpoint; because that checkpoint's weights are byte-identical to Experiment 1's (§2.4), the heatmaps themselves are unaffected by the threshold change — the threshold determines only which cases count as correctly classified in the breakdown at §3.5.

**3.3 Heatmap binarization.** IoU and Dice compare two binary sets, but Grad-CAM is continuous, so a threshold τ converts "how hot is this pixel" into "does the model claim this pixel": the claimed region is `P = {pixels with cam ≥ τ}`. Following Grad-CAM/weakly-supervised-localization convention, **τ = 0.50** — half of each map's own peak — is fixed *a priori*, before any result is inspected, so that no threshold is selected on the strength of the scores it produces. Because Grad-CAM normalizes each map by its own maximum, τ carries the same meaning on every image and every architecture.

**3.4 Metric computation.** For every image, with all pixel sets restricted to the validity mask:

- **IoU** = |P ∩ G| / |P ∪ G|, where G is the ground-truth ROI.
- **Dice coefficient** = 2|P ∩ G| / (|P| + |G|).
- **Pointing Game** — a hit is recorded if the heatmap's single maximum-activation valid pixel falls inside G; a tolerance variant (hit within 15 px of the mask) and the peak-to-mask distance are recorded alongside it.

Three threshold-free diagnostics are computed in parallel: **energy concentration** (the fraction of total heatmap mass falling inside the ROI), the **concentration ratio** (energy concentration divided by the chance rate, i.e. the ROI's share of valid pixels — a value of 3.0× means the model attends to the lesion three times more than a heatmap ignoring the image would), and pixel-level average precision and AUROC treating the heatmap as a per-pixel malignancy score.

**3.5 Aggregation and analysis.** Per-image scores are aggregated into dataset-level means — mean IoU, mean Dice, and overall Pointing Game accuracy — each reported with a percentile bootstrap 95% CI (2,000 resamples), overall and broken down by benign versus malignant cases and by correctly classified versus misclassified cases (the latter partition determined by the Experiment 2 threshold). All four architectures are scored on the *same* images in the *same* order, which is asserted programmatically, making the comparison paired.

**3.6 Statistical comparison of architectures.** Four architectures give six pairwise comparisons. The continuous, long-tailed IoU and Dice distributions are compared with the **Wilcoxon signed-rank test**, which assumes no normality — an assumption that is not safe at this sample size. All six p-values receive **Holm–Bonferroni correction** to control the family-wise error rate across the six hypotheses. Pointing Game accuracy is not significance-tested; it is reported descriptively with its bootstrap interval and read as directional support for the overlap-metric result rather than as independent evidence. Interpretability rank is then cross-tabulated against the classification rank from Experiment 1, to assess whether the most accurate architecture is also the most interpretable.

**3.7 Validity controls.** Because a localization metric that scores well for an untrained model would be measuring the dataset's geometry rather than anything the model learned, four controls are run *before* any headline number is accepted:

1. **Chance floor.** The full metric suite is recomputed for a randomly-initialized, untrained model of each architecture; scores must sit near the chance rate (concentration ratio ≈ 1.0×).
2. **Geometric round-trip.** A synthetic marker placed at a known fractional position in a source image must return to that same fractional position after the padding-and-resize round trip, verified across the real aspect ratios present in the dataset.
3. **Spatial correspondence.** A synthetic stride-32 network, for which the input-block-to-feature-cell mapping is known exactly, must produce its CAM peak at the input patch that was brightened — pinning down the height/width ordering in the upsampling step.
4. **Padding fraud check.** The percentage of images whose hottest pixel lands in the black padding is reported; a model keying on canvas geometry rather than tissue would show an elevated rate here.

Additionally, the full τ grid (0.05 to 0.95 in steps of 0.05) is swept and plotted as an appendix, establishing whether the architecture ranking is a property of the models or an artifact of the fixed τ = 0.50. Where the curves cross, the affected architectures are reported as tied.

---

## Summary of Data Gathered

| Experiment | Data gathered | Source | Statistical treatment |
|---|---|---|---|
| Shared (§0) | 384 × 640 image + mask cache; patient-grouped 80/10/10 splits | CBIS-DDSM DICOM + metadata CSVs | `StratifiedGroupKFold`, seed 42; zero-patient-overlap assertion |
| 1 — Baseline comparison | Accuracy, Sensitivity, Specificity, F1, AUC-ROC, training time per architecture; cached test-set probabilities | 4 training runs, identical recipe; 319-image test set | Bootstrap 95% CI (2,000 resamples); paired CI on Δ accuracy vs. dual-input teacher |
| 2 — Sensitivity optimization | Accuracy, Sensitivity, Specificity, F1 at 198 thresholds; two candidate operating points | Experiment 1's cached probabilities — no retraining, no new inference | Bootstrap 95% CI per candidate; CI-overlap test vs. default threshold |
| 3 — Explainability evaluation | Per-image IoU, Dice, Pointing Game hit, energy concentration, concentration ratio, pixel AP/AUROC | Grad-CAM on the Experiment 2 checkpoint vs. expert ROI masks, all 4 architectures | Bootstrap 95% CI; paired Wilcoxon signed-rank (IoU/Dice), Holm–Bonferroni corrected; PGA reported descriptively |
