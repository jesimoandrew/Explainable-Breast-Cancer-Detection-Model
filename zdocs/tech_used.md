# Technologies Used

This document lists the technologies, libraries, and tools used to build the explainable breast-cancer classification system described in [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md), and the reasoning behind each choice.

| Technology/Tool | Purpose | Justification |
|---|---|---|
| **Python 3.12** | Core programming language | The de facto standard for machine learning research; has the largest ecosystem of data science and deep learning libraries, so every other tool below plugs directly into it without translation layers. |
| **Jupyter Notebook** | Experiment authoring and reporting | Interleaves code, output, plots, and explanation in one document — well suited to research work where results must be inspected and iterated on step by step, and where the notebook itself becomes part of the thesis documentation. |
| **PyTorch** | Deep learning framework (model definition, training, autograd) | Its define-by-run autograd makes custom training loops, custom losses (e.g. knowledge distillation), and hook-based techniques like Grad-CAM straightforward to implement, without needing a separate graph-compilation step. |
| **torchvision** | Pretrained CNN backbones and image transforms | Ships ImageNet-pretrained DenseNet121, ResNet-50, EfficientNet-B2, and VGG-16 with a consistent API, avoiding the need to train four large backbones from scratch on a comparatively small medical dataset. |
| **pydicom** | Reading medical DICOM images | The mammograms are distributed in DICOM, the standard medical imaging format; pydicom is the standard Python library for decoding DICOM pixel data and metadata (including VOI LUT windowing). |
| **Pillow (PIL)** | Image loading, resizing, and format conversion | Used for every raster operation in the preprocessing pipeline (DICOM → PNG conversion, aspect-preserving resize, padding, grayscale/RGB conversion) — lightweight and sufficient without a heavier computer-vision dependency. |
| **OpenCV** | Auxiliary image processing | Available in the environment for pixel-level image operations that fall outside Pillow's scope. |
| **NumPy** | Numerical array operations | Backs nearly every array manipulation in the pipeline (pixel arrays, CAM heatmaps, metric computations) with vectorized, C-speed operations. |
| **pandas** | Tabular data handling | Used throughout data cleaning, patient-level splitting, and results aggregation to manage the CBIS-DDSM metadata as structured, filterable tables. |
| **scikit-learn** | Classical ML utilities and evaluation metrics | Supplies `StratifiedGroupKFold` for the patient-level, leakage-safe, label-stratified train/val/test split, and evaluation metrics such as ROC-AUC and average precision used to score both classification and Grad-CAM localization quality. |
| **SciPy** | Statistical hypothesis testing | Provides the Wilcoxon signed-rank test used to compare architectures' interpretability scores pairwise, with Holm–Bonferroni correction applied on top for the multiple-comparison problem. |
| **Matplotlib** | Plotting and figure generation | Used to render training curves, Grad-CAM overlay grids, and localization comparison figures included in the thesis. |
| **tqdm** | Progress reporting | Wraps long-running loops (image caching, training epochs, evaluation) with a progress bar, making multi-hour notebook runs easier to monitor. |
| **uv** | Python package and virtual-environment manager | Manages project dependencies and lockfile (`pyproject.toml` / `uv.lock`) reproducibly, including pinning the CUDA-specific PyTorch build required for GPU training. |
| **CUDA (via PyTorch cu128 build)** | GPU-accelerated training and inference | Training four CNN backbones at 384×640 resolution over multiple epochs is computationally expensive; GPU acceleration is what makes the experiment tractable in practice. |
| **Git** | Version control | Tracks the evolution of the preprocessing pipeline, model code, and experiment notebooks across iterations (e.g. the 224px → 384×640 caching redesign), and is the basis for the thesis's reproducibility trail. |

## Dataset

| Resource | Purpose | Justification |
|---|---|---|
| **CBIS-DDSM** (Curated Breast Imaging Subset of DDSM) | Source mammography dataset | A publicly available, expert-annotated mammography dataset with pathology labels (benign/malignant) and per-lesion ROI masks — the ROI masks are what make both knowledge distillation (teacher crop input) and quantitative Grad-CAM validation possible. |
