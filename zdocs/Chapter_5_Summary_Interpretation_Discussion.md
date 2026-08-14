# CHAPTER 5

## SUMMARY OF FINDINGS, PRESENTATION, INTERPRETATION, DISCUSSION OF RESULTS AND RECOMMENDATIONS

---

## 5.1 Introduction

This chapter presents the results of the system evaluation conducted across three sequential experiments on the CBIS-DDSM mammography dataset. Because the study is experimental deep-learning research rather than a survey-based system evaluation, the evaluation instrument is not a questionnaire administered to human evaluators but a set of **functionally grounded quantitative measurements** computed directly from model outputs on a held-out test set of 319 mammograms drawn from 142 patients, none of whom appear anywhere in training.

The evaluation focused on three dimensions, each corresponding to one experiment:

1. **Classification performance** — which of four pretrained CNN architectures best discriminates benign from malignant tissue under an identical training recipe (Experiment 1).
2. **Clinical operating point** — whether the decision threshold can be adjusted, without retraining, to favor sensitivity in a manner appropriate to cancer screening (Experiment 2).
3. **Explanation fidelity** — whether the Grad-CAM heatmaps the system produces actually localize the lesion, measured against radiologist-drawn ROI masks (Experiment 3).

The data collected from these experiments were analyzed, summarized, and interpreted to determine the overall quality and effectiveness of the developed system. All figures reported below are read directly from saved result files; none are re-derived or estimated. Every headline metric carries a percentile bootstrap 95% confidence interval (2,000 resamples), since at n = 319 a bare point estimate is not defensible on its own.

---

## 5.2 Summary of Findings

### 5.2.1 Classification Performance (Experiment 1)

All four architectures — DenseNet121, VGG-16, EfficientNet-B2, and ResNet-50 — were trained as single-input students under an identical configuration (384 × 640 input, batch size 8, seed 42, GAP pooling, distillation α = 0.5), so that any performance difference is attributable to the backbone rather than to the training regimen.

**DenseNet121 achieved the best classification performance**, with an accuracy of 0.7524 [0.705, 0.799] and an AUC-ROC of 0.8390 [0.796, 0.879]. All four architectures cleared the majority-class floor of 58.3%, confirming that each learned genuine discriminative structure rather than defaulting to the dominant class. However, **the four architectures' confidence intervals overlap substantially**, indicating that the ranking among them is not sharply resolved at this sample size.

The deployability check found that **no architecture showed a detectable accuracy loss from dropping the expert lesion crop at inference**: every paired confidence interval on Δ accuracy spans zero. This validates the single-input design on which the entire system rests.

DenseNet121's advantage carried a compute cost: it required 71.7 minutes to converge, roughly **twice** the training time of the other three architectures.

### 5.2.2 Clinical Operating Point (Experiment 2)

At its default 0.500 threshold, the selected model exhibited **sensitivity (72.5%) below its specificity (77.3%)** — the wrong direction for a screening application, where a missed malignancy is costlier than a false alarm. Of 319 test cases, the model produced 38 false negatives against 41 false positives.

A 198-point threshold sweep over the cached test-set probabilities identified two candidate operating points:

- **Crossover (t = 0.485)** raised sensitivity to 74.6% and reversed the sensitivity/specificity ordering at **no detectable accuracy cost** — its accuracy interval overlaps the default's entirely.
- **High-Sensitivity (t = 0.305)** raised sensitivity to **91.3% [0.861, 0.958]** while *also* improving accuracy to 0.7618, at the cost of specificity falling to 64.6%.

Both improvements were obtained **without retraining and without any new inference**, at negligible computational cost — the entire sweep, all bootstrap intervals, and both checkpoint re-saves complete in under a minute of CPU time.

### 5.2.3 Explanation Fidelity (Experiment 3)

Grad-CAM heatmaps were scored against expert ROI masks for all four architectures at a binarization threshold τ = 0.50 fixed *a priori*.

**All four architectures localize substantially better than chance.** Against a chance floor of 1.14% (the lesion's mean share of valid pixels), pointing-game accuracy ranged from 8.78% to 16.93% — a lift of 7.7× to 14.8× — and concentration ratios ranged from 2.74× to 5.92×.

**VGG-16 was significantly the most interpretable architecture**, leading on every metric and outperforming all three competitors on IoU and Dice after Holm–Bonferroni correction (adjusted p from 7.3 × 10⁻⁷ to 2.1 × 10⁻⁴). The remaining three architectures were largely indistinguishable from one another.

Two further patterns emerged. **Malignant cases are localized five to ten times better than benign cases** (VGG-16: 34.78% versus 3.31% pointing accuracy), which is expected given that heatmaps are always computed with respect to the malignant class. And **correctly classified cases are localized roughly two to three times better than misclassified ones**, indicating that explanation quality and prediction quality are positively related in aggregate.

### 5.2.4 Cross-Experiment Finding

The single most consequential result of the study emerges only when Experiments 1 and 3 are read together: **the best-performing classifier is the worst explainer.** DenseNet121 ranks first on accuracy and AUC but last on mean interpretability rank (3.50 of 4), while VGG-16 ranks second on accuracy but first on interpretability (mean rank 1.25). The two rankings are close to inverted.

---

## 5.3 Presentation of Data

Because the study reports continuous performance metrics rather than Likert-scale responses, results are interpreted against **established benchmark bands for diagnostic discrimination** rather than agreement categories. The AUC-ROC interpretation scale below follows the conventional criteria for discriminative ability:

| AUC-ROC Range | Interpretation |
|---|---|
| 0.90 – 1.00 | Outstanding discrimination |
| 0.80 – 0.90 | **Excellent / good discrimination** |
| 0.70 – 0.80 | Acceptable / fair discrimination |
| 0.60 – 0.70 | Poor discrimination |
| 0.50 – 0.60 | No discrimination (chance) |

### 5.3.1 Classification Performance by Architecture

**Table 5.1.** Classification performance of the four architectures on the held-out test set (n = 319), with percentile bootstrap 95% confidence intervals.

| Architecture | Accuracy [95% CI] | AUC-ROC [95% CI] | Sensitivity | Specificity | F1 | Interpretation |
|---|---|---|---|---|---|---|
| **DenseNet121** | **0.7524** [0.705, 0.799] | **0.8390** [0.796, 0.879] | 0.7246 | 0.7735 | 0.7168 | **Excellent** |
| VGG-16 | 0.7273 [0.677, 0.774] | 0.8328 [0.789, 0.873] | 0.7391 | 0.7182 | 0.7136 | Excellent |
| EfficientNet-B2 | 0.7147 [0.664, 0.765] | 0.8133 [0.767, 0.858] | 0.6957 | 0.7293 | 0.6966 | Excellent |
| ResNet-50 | 0.7022 [0.652, 0.749] | 0.7939 [0.744, 0.840] | 0.6884 | 0.7127 | 0.6849 | Acceptable |
| *Majority-class floor* | *0.5830* | *0.5000* | — | — | — | *Chance* |

**Table 5.2.** Compute cost and convergence by architecture.

| Architecture | Training time (min) | Best epoch | Epochs run | Relative cost |
|---|---|---|---|---|
| DenseNet121 | 71.7 | 19 | 27 | 2.1× |
| ResNet-50 | 37.7 | 20 | 28 | 1.1× |
| VGG-16 | 37.3 | 10 | 18 | 1.1× |
| EfficientNet-B2 | 34.4 | 17 | 25 | 1.0× |

### 5.3.2 Decision Threshold Optimization

**Table 5.3.** Performance at the default and two candidate operating points, DenseNet121, same test set. AUC-ROC is invariant across thresholds by construction.

| Operating point | Threshold | Accuracy [95% CI] | Sensitivity [95% CI] | Specificity [95% CI] | F1 | AUC-ROC |
|---|---|---|---|---|---|---|
| Main (default) | 0.500 | 0.7524 [0.705, 0.799] | 0.7246 | 0.7735 | 0.7168 | 0.8390 |
| Crossover | 0.485 | 0.7429 [0.696, 0.790] | 0.7464 [0.669, 0.816] | 0.7403 [0.677, 0.805] | 0.7153 | 0.8390 |
| **High-Sensitivity** | **0.305** | **0.7618** [0.718, 0.809] | **0.9130** [0.861, 0.958] | 0.6464 [0.577, 0.714] | **0.7683** | 0.8390 |

**Table 5.4.** Confusion matrix decomposition at the default threshold (t = 0.500), DenseNet121.

| | Predicted Malignant | Predicted Benign | Total |
|---|---|---|---|
| **Actually Malignant** | 100 (TP) | **38 (FN)** | 138 |
| **Actually Benign** | 41 (FP) | 140 (TN) | 181 |
| **Total** | 141 | 178 | 319 |

The 38 false negatives against 41 false positives is the imbalance Experiment 2 was designed to correct.

### 5.3.3 Explanation Fidelity by Architecture

**Table 5.5.** Localization metrics against expert ROI masks, τ = 0.50, n = 319, with bootstrap 95% confidence intervals.

| Architecture | IoU [95% CI] | Dice [95% CI] | Pointing Game [95% CI] | Lift over chance | Concentration ratio |
|---|---|---|---|---|---|
| **VGG-16** | **0.0761** [0.059, 0.095] | **0.1102** [0.088, 0.135] | **16.93%** [12.9, 21.0] | **14.8×** | **5.92×** |
| ResNet-50 | 0.0487 [0.036, 0.062] | 0.0752 [0.058, 0.094] | 9.09% [6.0, 12.5] | 8.0× | 4.26× |
| EfficientNet-B2 | 0.0464 [0.036, 0.057] | 0.0758 [0.060, 0.091] | 8.78% [6.0, 12.2] | 7.7× | 3.38× |
| DenseNet121 | 0.0351 [0.026, 0.044] | 0.0584 [0.045, 0.072] | 9.72% [6.6, 13.2] | 8.5× | 2.74× |
| *Chance floor* | — | — | *1.14%* | *1.0×* | *1.00×* |

**Table 5.6.** Localization by true class, showing the clinically meaningful malignant subgroup.

| Architecture | Benign (n = 181) Dice / PGA | Malignant (n = 138) Dice / PGA |
|---|---|---|
| **VGG-16** | 0.036 / 3.31% | **0.208 / 34.78%** |
| ResNet-50 | 0.022 / 1.10% | 0.146 / 19.57% |
| EfficientNet-B2 | 0.029 / 2.21% | 0.137 / 17.39% |
| DenseNet121 | 0.023 / 2.76% | 0.105 / 18.84% |

**Table 5.7.** The accuracy–interpretability inversion, same four architectures on the identical test split.

| Architecture | AUC-ROC | Accuracy rank | Dice | Interpretability rank |
|---|---|---|---|---|
| DenseNet121 | 0.8390 | **1** | 0.0584 | **4** |
| VGG-16 | 0.8328 | 2 | **0.1102** | **1** |
| EfficientNet-B2 | 0.8133 | 3 | 0.0758 | 3 |
| ResNet-50 | 0.7939 | 4 | 0.0752 | 2 |

---

## 5.4 Interpretation of Results

### 5.4.1 On Classification Performance

The system obtained an AUC-ROC of 0.8390 for its selected architecture, placing it in the **"excellent discrimination"** band and indicating that the model reliably ranks malignant cases above benign ones. Interpreted against the study's objectives, this confirms that a whole-image classifier trained on CBIS-DDSM without ROI cropping, radiomics fusion, or ensembling can achieve clinically meaningful discriminative ability.

The overlapping confidence intervals across all four architectures carry an important qualification: **DenseNet121's selection is defensible on point estimates but is not a statistically resolved victory.** A study reporting only the winning number would overstate the certainty of that choice. This matters directly for §5.4.4, where a competing selection criterion produces a different winner.

The null result on the deployability check is the study's quiet structural finding. It confirms that the system's central design constraint — that a clinician will only ever have the full mammogram — costs nothing detectable in accuracy. Without this, the single-input architecture would be an unjustified simplification rather than a validated design decision. Two limits apply: an interval spanning zero indicates *undetectability*, not equivalence, and the compared models differ in input resolution as well as in crop availability.

### 5.4.2 On the Clinical Operating Point

The default-threshold result — sensitivity below specificity — represents a genuine misalignment between the model's default behavior and its intended clinical use. A screening-support tool that misses 38 of 138 malignancies while raising 41 false alarms is optimizing the wrong error.

The High-Sensitivity threshold's 91.3% sensitivity means the system correctly flags roughly **nine in ten malignant cases**, against seven in ten at the default. In screening terms this is the difference between a tool that can be trusted to rule out disease and one that cannot. That it also *improved* accuracy (0.7618 versus 0.7524) is unusual and reflects the class distribution of the test set; the more typical expectation would be a sensitivity gain paid for in accuracy.

Two qualifications are essential to an honest reading. First, **specificity fell to 64.6%**, meaning roughly one benign case in three is flagged for review — an acceptable trade in a screening triage context, but a real operational cost. Second, the High-Sensitivity threshold was selected by maximizing accuracy over ~200 candidate cutoffs **on the same test set on which it is reported**, a mild form of test-set selection bias. Its point estimates should be read as an illustrative upper bound rather than a validated final choice. The Crossover threshold, whose definition is structural rather than performance-maximizing, is free of this concern and is the more conservative recommendation.

### 5.4.3 On Explanation Fidelity

The pointing-game results establish that **the system's explanations are genuinely informative rather than decorative**. At 14.8× the chance rate for the best architecture, and with a randomly-initialized control scoring at the chance floor, the heatmaps demonstrably reflect learned structure rather than dataset geometry.

The absolute IoU and Dice values are low, and their interpretation requires care. The lesion occupies a median of only 0.31% of the canvas, while Grad-CAM output is quantized to a 20 × 12 feature grid whose smallest expressible region is one ~32 × 32-pixel cell. A perfectly placed single-cell heatmap therefore caps out at a modest IoU. **These metrics penalize a resolution ceiling as much as a localization error**, which is why the threshold-free pointing game is the appropriate figure to lead with and why IoU and Dice should never be quoted without stating τ.

The class-conditional result — malignant cases localized five to ten times better than benign ones — should not be read as a deficiency. Heatmaps are computed with respect to the malignant class for every image, so on a benign case the model is being asked "what here looks malignant" about an image where the correct answer is *nothing*; a diffuse, low-scoring heatmap is the appropriate response. The malignant subgroup carries the clinically meaningful figures, and there VGG-16 places its peak inside the radiologist's ROI on better than **one case in three**.

The correctness-conditional result carries a practical warning. While explanation quality and prediction quality correlate in aggregate, the relationship is far from deterministic case by case: the system can classify a malignant case correctly while pointing hundreds of pixels away from the lesion. **A clinician cannot treat a plausible-looking heatmap as confirmation that the prediction was made for the right reason.**

### 5.4.4 On the Accuracy–Interpretability Inversion

This is the finding that most directly challenges the study's own architecture selection. Ranked by discrimination, DenseNet121 wins; ranked by explanation fidelity, it comes last, significantly behind VGG-16 on both overlap metrics after correction for multiple comparisons.

The accuracy VGG-16 surrenders is small — 2.51 percentage points of accuracy and 0.62 points of AUC, neither difference statistically resolved given the overlapping intervals in Table 5.1. What it gains is roughly **double the Dice and nearly double the pointing-game accuracy**.

For a system whose value proposition rests on a radiologist being able to *verify the model's reasoning*, an explanation that lands on the lesion one time in six is materially more useful than one that lands one time in ten. The result therefore constitutes a defensible case for revisiting the architecture selection under a combined accuracy-plus-interpretability criterion — stated here as a finding rather than acted upon, since re-running the selection falls outside the scope of the present evaluation.

---

## 5.5 Discussion

### 5.5.1 Overall System Evaluation

Taken together, the three experiments describe a system that performs **competently at classification, strongly at sensitivity once properly calibrated, and modestly but genuinely at explanation**.

The pipeline as a whole is internally coherent: each experiment consumes the artifacts of the one before it, and each result is validated against a control designed to make it falsifiable. The methodological posture throughout has been to test the measuring instrument before trusting its readings — patient-level leakage checks before reporting accuracy, weight-integrity verification before reporting threshold effects, and four separate validity controls (chance floor, geometric round-trip, spatial correspondence, padding fraud check) before reporting any localization score.

The system's headline claim is best stated as its sensitivity rather than its accuracy: **approximately 91% of malignant cases are correctly flagged, with explanations that localize the lesion at roughly fifteen times the chance rate.**

### 5.5.2 Comparison with Related Studies

The study's results align closely with realistic CBIS-DDSM benchmarks reported in the literature reviewed in Chapter 2.

**On classification.** CBIS-DDSM is a known-difficult dataset that consistently yields lower performance than curated alternatives such as INbreast. A MobileNet transfer-learning study reported **74.5% accuracy on CBIS-DDSM** while still outperforming AlexNet, VGG-16, GoogLeNet, and ResNet on the task; the present study's 75.24% sits essentially at that well-established baseline. Petrini et al. (2022) reported an AUC of **0.8483 under CBIS-DDSM's original training/test split** for an end-to-end EfficientNet two-view classifier — among the strongest results reported for the task — against the present study's 0.8390 for a **single-view** classifier. A subsequent multi-view study reported that a single-view patch-based benign-versus-malignant classifier reached an AUC of only **0.769**, which the present single-view result exceeds.

One methodological difference makes this comparison conservative rather than flattering: several of these benchmarks use CBIS-DDSM's *original* published train/test split, which the present study found to leak patients between partitions and therefore rebuilt from scratch using patient-grouped stratification. The present results are obtained on a **stricter, leak-free split** and are correspondingly harder-won.

**On sensitivity.** The literature establishes that sensitivity in the high 70s to low 90s is a realistic operating range for mammographic classification; a DDSM mass segmentation-and-classification framework reported 77.89% sensitivity at 78.38% accuracy. The present study's **91.3% sensitivity sits at the upper end** of that range. The Philippine tuberculosis screening deployment of Marquez et al. (2025) — 95.6% pseudo-sensitivity against 28.1% pseudo-specificity — offers a local precedent for precisely the trade this study makes: a screening tool is deliberately tuned to rule out disease, accepting reduced specificity because missing true disease is the more consequential error. Against that benchmark, the present system's 64.6% specificity at 91.3% sensitivity is a considerably more balanced operating point than an accepted clinical deployment.

**On explainability.** The XAI literature (Patrício et al., 2023) holds that clinical adoption depends on interpretability rather than accuracy alone, and Suara et al. (2024) specifically examine whether Grad-CAM is explainable in medical images, affirming its interpretive value while cautioning about its limitations. The present study contributes a quantitative answer to exactly that question on mammography: Grad-CAM explanations are informative well above chance, but their absolute overlap with expert annotations is low and bounded by feature-map resolution. Where the literature reports Grad-CAM mammography frameworks achieving 99%-plus accuracy, those results are obtained on curated or heavily augmented datasets and should not be treated as representative baselines for whole-image CBIS-DDSM classification.

**On the accuracy–explainability relationship.** Chapter 2 argues that explanation-oriented design choices — preserving spatial resolution, avoiding ROI-only inputs, training on whole images, prioritizing recall — can reduce headline accuracy while producing more trustworthy outputs. The present study's inversion finding (§5.4.4) supplies direct empirical evidence for that position from *within* a single controlled experiment: holding data, recipe, and evaluation constant, the architecture with the highest AUC produced the least faithful explanations.

### 5.5.3 Implications

**For system design.** The deployability result validates the study's central architectural commitment: a model requiring only the full mammogram matches a model given expert lesion crops, within measurement noise. This makes the system deployable in a setting where no prior annotation exists — which is the only setting that matters clinically.

**For clinical relevance.** Chapter 2 documents the Philippine context: the highest breast cancer incidence in Asia, a screening rate as low as 2.22%, 53% of cases diagnosed at Stage 3 or 4, and five-year survival of approximately 44%. In that context, a sensitivity-prioritizing triage tool that flags nine in ten malignancies addresses the specific failure mode — late detection — that drives poor outcomes. A system optimized for accuracy at the default threshold would have missed 38 of 138 malignant cases; the recalibrated system misses 12.

**For explainable AI in medical imaging.** The study demonstrates that Grad-CAM explanations can be evaluated *functionally* — against annotations already present in the dataset — rather than solely through expert visual review. This is a cheaper, more reproducible, and less subjective evaluation path, and one that surfaced a finding (the accuracy–interpretability inversion) that a purely qualitative review would likely have missed.

### 5.5.4 Limitations

Several limitations qualify the findings and are stated explicitly rather than deferred:

1. **Test-set selection bias on the High-Sensitivity threshold.** Selected by argmax over ~200 cutoffs on the same test set on which it is reported. The Crossover threshold is free of this concern.
2. **Statistical power.** At n = 319, the four architectures' confidence intervals overlap substantially; the architecture ranking is not sharply resolved.
3. **Resolution confound in the deployability comparison.** The single-input models run at 384 × 640 while the dual-input reference was trained at 224², so that comparison holds the test split constant but not the input size.
4. **Localization resolution ceiling.** Grad-CAM's 20 × 12 feature grid bounds achievable IoU and Dice irrespective of how well the model actually localizes.
5. **No inference-time benchmarking.** Per-image latency, throughput, and serving memory footprint were not formally measured in any experiment.
6. **Single dataset, single modality.** All results are obtained on CBIS-DDSM; generalization to other mammography datasets or to digital breast tomosynthesis is untested.
7. **Software quality not evaluated.** The demonstration application was not assessed under a software-quality instrument such as ISO/IEC 25010; that evaluation would require a separate questionnaire administered to system evaluators and is outside the scope of the three experiments reported here.

### 5.5.5 Recommendations for Improvement

Based on the observed results and limitations, the following are recommended:

1. **Re-run architecture selection under a combined criterion.** Given the inversion finding, the selection should weigh explanation fidelity alongside AUC. On present evidence VGG-16 is the stronger candidate for a decision-support deployment.
2. **Select the operating threshold on the validation split.** Caching validation-set probabilities and choosing the threshold there, then evaluating once on test, would eliminate the selection bias attached to the High-Sensitivity figure at negligible cost.
3. **Raise Grad-CAM resolution.** Reducing the backbone's final downsampling stride, or adopting a higher-resolution explanation method, would relax the ceiling that currently bounds IoU and Dice.
4. **Benchmark inference cost.** Per-image latency and memory footprint should be measured before any deployment claim is made.
5. **Evaluate on an external dataset.** Validation on INbreast or a local Philippine mammography cohort would test whether the findings generalize beyond CBIS-DDSM.
6. **Explore radiology-specific preprocessing.** The current pipeline deliberately applies only geometric resizing; DICOM windowing or CLAHE contrast enhancement remains an untested avenue for accuracy gains.
7. **Conduct a clinician-facing evaluation.** The functional metrics establish that heatmaps localize above chance; whether radiologists find them *useful* is a separate question requiring a separate instrument.

---

## 5.6 Summary

This chapter presented, interpreted, and discussed the results of three sequential experiments evaluating an explainable CNN-based mammography classification system on the CBIS-DDSM dataset.

**Strengths.** The system achieves excellent discriminative ability (AUC 0.8390) using only the full mammogram, matching or exceeding realistic published CBIS-DDSM benchmarks despite being evaluated on a stricter, leak-free patient-grouped split. Post-hoc threshold calibration raises sensitivity to 91.3% at no retraining cost — a figure at the upper end of what the literature reports for this task and directly aligned with the clinical priority of not missing malignancies. Grad-CAM explanations localize lesions at up to 14.8 times the chance rate, and this claim is supported by four independent validity controls rather than asserted.

**Areas for improvement.** Absolute localization overlap remains low, bounded by Grad-CAM's feature-map resolution. The gain in sensitivity is paid for in specificity, and the threshold that produces the headline sensitivity figure carries a disclosed selection bias. Most significantly, the architecture selected for its classification performance proved to be the least faithful explainer of the four — a tension the study surfaces but does not resolve.

**Readiness.** The system is validated as a research prototype and as a demonstration of a functionally grounded explainability evaluation methodology. It is **not** ready for clinical deployment: external validation, inference-cost benchmarking, a clinician-facing usability evaluation, and resolution of the architecture-selection tension all remain outstanding.

The results presented in this chapter serve as the basis for the conclusions and recommendations presented in the succeeding chapter.
