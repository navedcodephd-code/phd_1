# NeuroFM-AX

**An Anatomy-Conditioned Cross-Modal Foundation Model with Low-Rank Adaptation for Calibrated Differential Diagnosis of Parkinson's Disease**

> Naved Ahmad, Ihtiram Raza Khan, Suraiya Parveen, Siddhartha Sankar Biswas  
> Jamia Hamdard University, New Delhi, India

---

## Overview

NeuroFM-AX is a two-stage framework for four-class differential diagnosis of Parkinson's Disease (HC / PD / Prodromal / SWEDD) on PPMI brain MRI:

- **Stage 1** – Anatomy-conditioned cross-modal pretraining (T1, T2-FLAIR, NM-MRI, DAT-SPECT) via Masked Volume Modeling  
- **Stage 2** – LoRA fine-tuning on the PPMI four-class task with a supervised contrastive head  

**Key results on held-out test set (N=75):** Accuracy 94.67% [89.33, 98.67], Macro-AUC 0.992, ECE 0.028.

---

## Repository Structure

```
neurofm_ax/
├── configs/                  # YAML configuration files
│   ├── pretrain.yaml
│   └── finetune.yaml
├── data/
│   ├── splits/               # Subject-level train/val/test split files (seed-42, 17, 2024)
│   └── dataset.py            # PPMI dataset class
├── preprocessing/
│   ├── pipeline.py           # Full preprocessing pipeline (FSL-BET → N4 → MNI → CLAHE)
│   └── verify_quality.py     # PSNR / SSIM / MSE quality checks
├── models/
│   ├── tokenizer.py          # Anatomy-aware tokenizer (ROI + global tokens)
│   ├── vit3d.py              # 3D Vision Transformer backbone
│   ├── lora.py               # LoRA adapter injection
│   ├── decoder.py            # Cross-modal MVM decoder
│   └── heads.py              # Classification + contrastive heads
├── training/
│   ├── pretrain.py           # Stage 1 pretraining loop
│   └── finetune.py           # Stage 2 LoRA fine-tuning loop
├── baselines/
│   ├── lenet5.py
│   ├── unet_cls.py
│   ├── resnet3d.py
│   ├── attention_lunet.py
│   ├── scratch_vit.py
│   ├── convkan.py
│   ├── swin_classifier.py
│   └── full_finetune.py
├── evaluation/
│   ├── metrics.py            # All metrics from integer confusion matrix
│   ├── calibration.py        # Temperature scaling + ECE/MCE/Brier
│   ├── bootstrap.py          # Bootstrap CI computation
│   └── statistical_tests.py  # Paired bootstrap significance tests
├── scripts/
│   ├── run_pretrain.sh
│   ├── run_finetune.sh
│   └── run_eval.sh
└── utils/
    ├── checkpoint.py         # Checkpoint selection rules
    ├── losses.py             # CE + supervised contrastive loss
    └── reproducibility.py    # Seed control helpers
```

---

## Installation

```bash
git clone https://github.com/<org>/neurofm-ax.git
cd neurofm-ax
conda create -n neurofmax python=3.10
conda activate neurofmax
pip install -r requirements.txt
```

**Requirements:** PyTorch 2.3, monai, nibabel, antspy, nilearn, scikit-learn, scipy, matplotlib, pyyaml.

---

## Data Access

All data are publicly available under their respective data-use agreements:

| Source | URL | Used for |
|--------|-----|---------|
| PPMI | ppmi-info.org | Fine-tuning (HC/PD/Prodromal/SWEDD) |
| UK Biobank | ukbiobank.ac.uk | Pretraining T1/FLAIR |
| ADNI | adni.loni.usc.edu | Pretraining T1/FLAIR |
| OASIS-3 | oasis-brains.org | Pretraining T1/FLAIR |
| IXI | brain-development.org/ixi-dataset | Pretraining T1 |
| NM-MRI collection | (multi-site, see paper) | Pretraining NM-MRI |

Subject-level split files for the 418-subject PPMI cohort (274/69/75 train/val/test, stratified by class, sex, site) are in `data/splits/`.

---

## Reproducibility

Table 7 in the paper reports **10-fold CV means under seed 42 only**. Seeds 17 and 2024 are provided as a stability check (SD ≤ 0.7 pp across seeds for every method).

```bash
# Reproduce seed-42 fine-tuning CV
python training/finetune.py --config configs/finetune.yaml --seed 42 --cv

# Reproduce three-seed stability check
for SEED in 42 17 2024; do
  python training/finetune.py --config configs/finetune.yaml --seed $SEED --cv
done
```

---

## Evaluation

```bash
# Evaluate held-out test set (run once, at the end)
python evaluation/metrics.py --checkpoint checkpoints/best_seed42.pt --split test

# Bootstrap 95% CI (B=1000)
python evaluation/bootstrap.py --checkpoint checkpoints/best_seed42.pt --n_bootstrap 1000

# Paired bootstrap significance test vs. baseline
python evaluation/statistical_tests.py \
  --model_preds preds/neurofmax_cv.npy \
  --baseline_preds preds/swinclassifier_cv.npy \
  --n_bootstrap 1000
```

---

## Citation

```bibtex
@article{ahmad2025neurofmax,
  title   = {NeuroFM-AX: An Anatomy-Conditioned Cross-Modal Foundation Model
             with Low-Rank Adaptation for Calibrated Differential Diagnosis
             of Parkinson's Disease},
  author  = {Ahmad, Naved and Khan, Ihtiram Raza and Parveen, Suraiya
             and Biswas, Siddhartha Sankar},
  journal = {(under review)},
  year    = {2025}
}
```

---

## License

Code: MIT License. Data are subject to the terms of the respective data-use agreements listed above.
