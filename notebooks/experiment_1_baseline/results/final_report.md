# Final report: hi-res single-input vs the original architectures

Arm `control` / pooling `gap` / alpha `0.5`, input 384x640, batch 8, seed 42.

## Controlled comparison

Both models scored on the identical 319-row hi-res test split. The dual-input numbers are recomputed here from the cached distillation logits rather than copied from the original CSV, because the original ran on a different split (see caveat).

| architecture | hi-res single-input acc [95% CI] | AUC [95% CI] | dual-input acc | dual-input AUC | delta acc [paired 95% CI] | McNemar p |
|---|---|---|---|---|---|---|
| DenseNet121 | 0.7524 [0.705-0.799] | 0.8390 [0.796-0.879] | 0.7868 | 0.8655 | -0.0345 [-0.088, +0.019] | 0.254 |
| VGG-16 | 0.7273 [0.677-0.774] | 0.8328 [0.789-0.873] | 0.7524 | 0.8402 | -0.0251 [-0.078, +0.034] | 0.445 |
| EfficientNet-B2 | 0.7147 [0.664-0.765] | 0.8133 [0.767-0.858] | 0.7743 | 0.8515 | -0.0596 [-0.122, +0.003] | 0.064 |
| ResNet-50 | 0.7022 [0.652-0.749] | 0.7939 [0.744-0.840] | 0.7210 | 0.8015 | -0.0188 [-0.075, +0.038] | 0.598 |

## Reading this

- **Best deployable model: DenseNet121**, 0.7524 accuracy / 0.8390 AUC, using only the full mammogram at inference.
- Hi-res single-input **beats** its dual-input counterpart for: (none).
- No significant difference (p >= 0.05) for: DenseNet121, VGG-16, EfficientNet-B2, ResNet-50. A null result matters here: it means dropping the ground-truth-derived crop costs nothing detectable at this sample size, which is the deployability argument.
- McNemar is paired and exact; only the disagreeing samples carry information. Bootstrap CIs are 2000 resamples.

## Caveat on the historical columns

The originally published CSVs were scored on different test splits -- the hi-res cache dropped 4 rows whose ROI mask dimensions genuinely disagreed with the scan. Row counts: 
  - `dual_input` -- inference inputs: full + GT crop; deployable: False; source `architecture_comparison_test_metrics.csv`
  - `three_input` -- inference inputs: full + crop + mask; deployable: False; source `three_input_ablation_test_metrics.csv`
  - `single_224` -- inference inputs: full only; deployable: True; source `single_input_distillation_ablation_test_metrics.csv`

So the historical columns are **context, not a controlled comparison**. Only the controlled table above supports a statistical claim.
