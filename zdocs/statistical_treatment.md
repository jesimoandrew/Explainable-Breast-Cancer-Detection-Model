# Statistical Treatment of Data

The data collected in this study are analyzed using quantitative statistical methods to provide an objective, data-driven assessment across two dimensions: **algorithmic accuracy** (Experiments 1 and 2) and **visual interpretability** (Experiment 3). All statistics are computed programmatically from saved model outputs rather than transcribed by hand, and every analysis is reproducible from the cached prediction files.

No human raters, expert panels, or survey instruments are involved at any stage. Interpretability is evaluated *functionally* — against the expert ROI masks already distributed with CBIS-DDSM — so it is treated with the same quantitative machinery as classification performance.

---

## 0. Conventions Common to All Three Experiments

These settings are fixed once and applied identically throughout, so that no result depends on an analysis choice made after seeing the data.

| Convention | Value | Rationale |
|---|---|---|
| Test set | 319 images, held out, patient-disjoint | Identical across all three experiments, enabling paired comparison |
| Random seed | 42 | Fixed across `random`, `numpy`, and `torch` for reproducibility |
| Confidence intervals | Percentile bootstrap, **2,000 resamples**, 2.5th/97.5th percentiles | At n = 319 the standard error on a proportion is ≈ 2.6 pp; a bare point estimate is not defensible |
| Significance level | α = 0.05 | |
| Multiple-comparison control | Holm–Bonferroni step-down | Uniformly more powerful than Bonferroni at the same family-wise error rate |
| Libraries | `scikit-learn` (metrics), `scipy.stats` (hypothesis tests), `numpy` (bootstrap resampling) | |

**Percentile bootstrap procedure.** For a per-image metric vector **v** of length *n*, draw 2,000 resamples of size *n* with replacement, compute the mean of each, and report the 2.5th and 97.5th percentiles of that distribution as the 95% CI. This is used for every headline figure in all three experiments because it makes no normality assumption — appropriate for bounded, skewed quantities such as IoU and for proportions such as sensitivity.

**Interpreting a confidence interval.** An interval that includes zero (for a difference) or that overlaps another interval is reported as *no detectable difference*. This is explicitly weaker than a claim of equivalence: it states that the available data cannot resolve a difference at this sample size, not that no difference exists.

---

## Experiment 1: Baseline Architecture Comparison

The diagnostic performance of each CNN architecture is treated using standard machine-learning metrics computed from the model's predictions on the held-out test set.

### 1.1 Classification Metrics

Accuracy, sensitivity, specificity, F1-score, and AUC-ROC are computed with `scikit-learn` from each model's test-set predictions. A confusion matrix is generated to display the distribution of correct and incorrect classifications, decomposed into true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN):

```
Accuracy     = (TP + TN) / (TP + TN + FP + FN)
Sensitivity  = TP / (TP + FN)          (recall; malignant cases correctly identified)
Specificity  = TN / (TN + FP)          (benign cases correctly identified)
Precision    = TP / (TP + FP)
F1-Score     = 2 × (Precision × Sensitivity) / (Precision + Sensitivity)
```

Sensitivity and specificity are reported separately rather than folded into accuracy alone, because the two carry asymmetric clinical cost: in a screening context a false negative (a missed malignancy) is more consequential than a false positive (an unnecessary recall). This asymmetry is what Experiment 2 acts on.

### 1.2 ROC Curve Analysis

A receiver operating characteristic curve is plotted for each architecture, tracing sensitivity against (1 − specificity) as the classification threshold varies across its full range. The area under this curve (**AUC-ROC**) is reported as a single scalar summarizing the model's ranking ability independent of any particular cutoff — that is, the probability that a randomly chosen malignant case receives a higher malignancy score than a randomly chosen benign case.

Because AUC-ROC is threshold-independent by construction, it is invariant to the threshold adjustment performed in Experiment 2 and serves as the stable basis for architecture selection.

### 1.3 Loss and Accuracy Curves

Training and validation loss, accuracy, and ROC-AUC are recorded per epoch and plotted to monitor convergence and detect overfitting or underfitting. Divergence between the training and validation curves indicates overfitting; the epoch at which validation loss reaches its minimum determines the saved checkpoint, and training is halted by early stopping after 8 epochs without improvement. Per-epoch histories are saved to `history_*.csv`, one file per architecture.

### 1.4 Interval Estimation

Accuracy and AUC-ROC are each reported with a percentile bootstrap 95% CI (2,000 resamples). Architecture selection is made on the point estimates, but the intervals are reported alongside so that the reader can see the degree to which the four architectures' performance ranges overlap — which they substantially do at this sample size.

### 1.5 Paired Comparison Against the Dual-Input Upper Bound

For each architecture, the deployable single-input model is compared against its non-deployable dual-input teacher on the identical test split. The difference in accuracy is expressed as a **paired bootstrap 95% confidence interval** on Δ accuracy (2,000 resamples), where each resample draws the same image indices for both models so that per-image difficulty cancels out.

An interval spanning zero indicates that the accuracy cost of dropping the expert lesion crop at inference is not detectable at this sample size. Two caveats are attached to the reading of this result: (a) undetectability is not equivalence — the interval's width still admits a meaningful loss; and (b) the two compared models differ in input resolution (384 × 640 versus 224²) as well as in crop availability, so the comparison holds the test split constant but not the input size.

---

## Experiment 2: Sensitivity Optimization

Experiment 2 performs no new inference. All statistics are recomputed from Experiment 1's cached test-set probability vector under varying decision cutoffs, so the model's ranking ability — and therefore its AUC-ROC — is fixed throughout.

### 2.1 Threshold Sweep

The decision threshold *t* is swept across **198 fixed steps from 0.010 to 0.995**. At each cutoff, predictions are re-derived as `ŷ = 1 if p ≥ t else 0` and Accuracy, Sensitivity, Specificity, and F1-Score are recomputed from the resulting confusion matrix using the formulas in §1.1. The result is a four-curve trace over *t*, saved to `sensitivity_threshold_sweep.csv` and plotted as the sensitivity/specificity crossover figure.

### 2.2 Candidate Selection Rules

Two candidate operating points are identified by **deterministic rules applied to the sweep**, not by manual inspection, so that the selection is reproducible:

- **Crossover threshold** — the largest *t* at which sensitivity still exceeds specificity. This is the minimal departure from the default cutoff that reverses the sensitivity/specificity ordering.
- **High-Sensitivity threshold** — the *t* maximizing accuracy among all thresholds satisfying sensitivity > specificity. In this sweep it coincides with the maximizer of **Youden's J statistic**:

```
J = Sensitivity + Specificity − 1
```

Youden's J is the standard single-number summary for choosing an operating point on an ROC curve, weighting the two error types equally; it is reported as corroboration that the selected cutoff is not an artifact of the accuracy criterion alone.

### 2.3 Interval Estimation and Comparison

Each candidate threshold's Accuracy, Sensitivity, and Specificity are reported with a percentile bootstrap 95% CI (2,000 resamples per metric per candidate). Candidates are compared against the default *t* = 0.500 by **confidence-interval overlap**: a candidate whose accuracy interval overlaps the default's is treated as imposing no detectable accuracy cost, and the sensitivity gain is therefore obtained without a measurable trade.

### 2.4 Disclosure of Selection Bias

The High-Sensitivity threshold is selected by argmax over ~200 candidate cutoffs evaluated on the *same* test set on which its performance is subsequently reported. This constitutes **test-set selection bias**, and its point estimates are accordingly presented as an illustrative upper bound rather than as a validated operating point. The methodologically clean alternative — selecting the threshold on the validation split and evaluating once on test — is identified as a limitation rather than silently omitted. The Crossover threshold is not subject to this concern, since its definition is structural (the sensitivity/specificity crossing) rather than performance-maximizing.

### 2.5 Weight-Integrity Verification

Because both candidate thresholds are distributed as standalone checkpoints, each re-saved checkpoint's `model_state_dict` is verified tensor-by-tensor against the source checkpoint (`torch.equal` over every key). This confirms that the reported metric changes are attributable solely to the decision cutoff and not to any alteration of the trained weights.

---

## Experiment 3: Explainability Evaluation

Grad-CAM heatmaps are scored against the expert ROI masks distributed with CBIS-DDSM. All pixel sets are restricted to the **validity mask** — the non-padding region of the 384 × 640 canvas — because 7–12% of each cached image is black letterbox padding, where an activation peak would be meaningless.

### 3.1 Binarization Threshold

IoU and Dice compare two binary sets, but Grad-CAM produces a continuous heatmap. A threshold τ converts the heatmap into a claimed region `P = {pixels with cam ≥ τ}`. Following Grad-CAM and weakly-supervised-localization convention, **τ = 0.50** — half of each map's own peak — is fixed *a priori*, before any score is inspected. Because Grad-CAM normalizes every map by its own maximum, τ carries identical meaning across images and architectures. τ is a component of the metric definition, not a tuned parameter.

### 3.2 Overlap Metrics

For each image, writing *P* for the claimed region and *G* for the ground-truth ROI:

```
IoU   =  |P ∩ G| / |P ∪ G|
Dice  =  2 |P ∩ G| / (|P| + |G|)
```

These are reported together by convention, but they are **one measurement expressed two ways** — related deterministically by `Dice = 2·IoU / (1 + IoU)` — and are therefore not treated as two agreeing pieces of evidence.

### 3.3 Pointing Game Accuracy

A threshold-free measure: a **hit** is recorded if the heatmap's single maximum-activation valid pixel falls inside *G*. Dataset-level accuracy is the proportion of hits. A 15-pixel-tolerance variant and the Euclidean peak-to-mask distance are recorded alongside it as supplementary diagnostics.

Pointing Game accuracy is reported **descriptively**, with its bootstrap interval; no significance test is applied to it in this study. Its ordering is read as directional support for the overlap-metric results rather than as independent statistical evidence.

### 3.4 Threshold-Free Diagnostics

Three further quantities are computed per image, none requiring a choice of τ:

```
Energy concentration  =  Σ cam[G ∩ valid] / Σ cam[valid]
Chance rate           =  |G ∩ valid| / |valid|
Concentration ratio   =  Energy concentration / Chance rate
```

The **concentration ratio** is the primary threshold-free indicator: a value of 3.0 means the heatmap places three times more of its total activation mass on the lesion than an image-blind heatmap would by area alone. Pixel-level **average precision** and **AUROC** are additionally computed by treating the heatmap as a per-pixel malignancy score and the mask as ground truth.

### 3.5 Comparison Against the Chance Floor

Absolute IoU and Dice values are low in this setting for a geometric reason rather than a modeling one: the lesion occupies roughly 1–2.5% of valid pixels, while the Grad-CAM heatmap is quantized to a 20 × 12 feature grid, so the smallest expressible region is one ~32 × 32-pixel cell. Every metric is therefore interpreted **relative to its chance floor**, not against 1.0:

- The pointing-game chance rate is the mean lesion coverage (**1.14%**).
- The concentration-ratio chance value is **1.0×** by construction.
- As an empirical control, the full metric suite is recomputed for a **randomly-initialized, untrained** model of each architecture, which should — and does — score near these floors, confirming that the metric measures learned behavior rather than dataset geometry.

### 3.6 Paired Significance Testing

All four architectures are scored on the identical images in the identical order (asserted programmatically), so every architecture comparison is **paired**. Pairing is materially more powerful than comparing overlapping confidence intervals, because per-image difficulty cancels out.

The continuous IoU and Dice distributions are bounded below at zero and long-tailed, so normality is not assumed at any sample size. They are compared with the **Wilcoxon signed-rank test**, a paired non-parametric procedure operating on the signed ranks of per-image differences.

Four architectures yield six pairwise comparisons. Uncorrected, the probability of at least one spurious significant result would be substantially inflated, so all six p-values receive **Holm–Bonferroni step-down correction**: p-values are sorted ascending and the *i*-th receives an adjusted value of

```
p_adj(i) = max over j ≤ i of  (m − j + 1) × p(j),     capped at 1.0
```

where *m* = 6. Adjusted values are reported and compared against α = 0.05.

### 3.7 Subgroup Analysis

Per-image scores are aggregated into dataset-level means, each with a percentile bootstrap 95% CI, reported overall and broken down along two partitions:

- **By true class** (benign vs. malignant). Because heatmaps are always computed with respect to the malignant class, benign cases are being asked "what here looks malignant" about images where the correct answer is *nothing*; a diffuse, low-scoring heatmap is the appropriate response rather than a failure. The malignant subgroup therefore carries the clinically meaningful figures.
- **By classification correctness** (correct vs. misclassified, partitioned by the Experiment 2 threshold). This tests whether explanation quality and prediction quality are related.

### 3.8 Sensitivity Analysis on τ

Because τ was fixed by convention rather than tuned, its influence is verified rather than assumed. The full grid τ ∈ {0.05, 0.10, …, 0.95} is swept and the mean IoU and Dice for all four architectures are plotted across it. If the vertical ordering of the curves is preserved across the range, the architecture ranking is a property of the models rather than of the threshold choice, and the single reported value may be quoted without qualification. Where curves cross, the affected architectures are reported as tied.

### 3.9 Instrument-Validity Controls

Because a localization metric that scores well for an untrained model would be measuring the dataset rather than the model, four controls are executed *before* any headline figure is accepted:

1. **Chance floor** (§3.5) — random-initialization control per architecture.
2. **Geometric round-trip** — a synthetic marker at a known fractional position must return to that position after the padding-and-resize round trip; worst relative error must fall below 0.02.
3. **Spatial correspondence** — a synthetic stride-32 network, whose input-block-to-feature-cell mapping is known exactly, must place its CAM peak at the brightened input patch, pinning down height/width ordering in the upsampling step.
4. **Padding fraud check** — the proportion of images whose hottest pixel lands in black padding is reported; a model keying on canvas geometry rather than tissue would show an elevated rate.

A fifth check verifies the optimized τ-sweep implementation (a single `argsort` plus vectorized `searchsorted`) against the naive re-thresholding implementation, since a silent disagreement would corrupt every IoU and Dice reported.

---

## Summary of Statistical Treatment

| Experiment | Descriptive statistics | Interval estimation | Inferential testing |
|---|---|---|---|
| **1 — Baseline comparison** | Accuracy, Sensitivity, Specificity, F1, AUC-ROC, confusion matrix, ROC curves, per-epoch loss/accuracy curves | Percentile bootstrap 95% CI (2,000 resamples) on Accuracy and AUC-ROC | Paired bootstrap 95% CI on Δ accuracy vs. the dual-input upper bound |
| **2 — Sensitivity optimization** | Accuracy, Sensitivity, Specificity, F1 across 198 thresholds; Youden's J | Percentile bootstrap 95% CI per candidate threshold | CI-overlap comparison against the default cutoff; selection bias disclosed |
| **3 — Explainability evaluation** | Per-image IoU, Dice, Pointing Game accuracy, energy concentration, concentration ratio, pixel AP/AUROC; subgroup breakdowns | Percentile bootstrap 95% CI on all headline means | Paired Wilcoxon signed-rank on IoU/Dice, Holm–Bonferroni corrected across six pairs; Pointing Game descriptive only |

**Scope note.** This chapter treats algorithmic accuracy and visual interpretability. A third dimension — **software quality** of the deployed application (e.g. usability or an ISO/IEC 25010-based evaluation) — is not covered by Experiments 1–3 and would require a separate instrument and its own statistical treatment if included in the study.
