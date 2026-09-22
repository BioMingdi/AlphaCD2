# AlphaCD2

**AlphaCD2** is a sequence-based computational framework for predicting and prioritizing cytidine deaminase (CD) variants for base-editing applications.

AlphaCD2 combines **ESM-C protein representations** with a **BiLSTM activity predictor** to estimate on-target editing activity. In addition, auxiliary models predict:

* cytidine motif preference;
* editing-window boundaries; and
* sgRNA-independent off-target activity.

AlphaCD2 operates directly on protein sequences and does **not require experimentally determined structures, structure prediction, structural templates, or structural alignment**.

The framework can also be combined with protein language models such as **ProGen2** for language-model-guided generation and prioritization of novel cytidine deaminase variants.

---

## Overview

The AlphaCD2 prediction workflow is:

```text
Protein sequence
      |
      v
ESM-C 600M representation
      |
      +----------------------> BiLSTM activity predictor
      |                                 |
      |                                 v
      |                       On-target activity
      |
      +----------------------> Auxiliary predictor
                                        |
                                        +-- Motif preference
                                        +-- Editing window
                                        +-- sgRNA-independent off-target activity
```

For protein-design applications, AlphaCD2 can be incorporated into a language-model-guided workflow:

```text
CD sequence dataset
        |
        v
Protein language-model fine-tuning
        |
        v
Generated CD variants
        |
        v
AlphaCD2 prediction
        |
        v
Candidate prioritization
        |
        v
Experimental validation
```

---

## Repository contents

```text
AlphaCD2/
├── AlphaCD2_predict.py
├── AlphaCD2_ontarget.py
├── alphacd2_seed42_model.py
├── bilstm_manifest.json
│
├── seed_42/
│   ├── bilstm_checkpoint.pt
│   └── scaler.pkl
├── seed_43/
│   ├── bilstm_checkpoint.pt
│   └── scaler.pkl
├── seed_44/
│   ├── bilstm_checkpoint.pt
│   └── scaler.pkl
│
├── auxiliary_metrics.pt
├── auxiliary_metrics.input_scaler.pkl
│
├── AlphaCD2_predict.sh
├── AlphaCD2_generation_data_preparing.sh
├── AlphaCD2_generation_Progen2_finetuning.sh
│
├── environment.yml
├── test.txt
├── LICENSE
└── README.md
```

### Main files

| File                                        | Description                                                             |
| ------------------------------------------- | ----------------------------------------------------------------------- |
| `AlphaCD2_predict.py`                       | Unified AlphaCD2 prediction pipeline                                    |
| `AlphaCD2_ontarget.py`                      | ESM-C embedding generation and BiLSTM on-target prediction              |
| `alphacd2_seed42_model.py`                  | Model definitions required for loading the released AlphaCD2 predictors |
| `bilstm_manifest.json`                      | Configuration of the released full-data BiLSTM models                   |
| `seed_42/`, `seed_43/`, `seed_44/`          | Pretrained BiLSTM checkpoints and preprocessing scalers                 |
| `auxiliary_metrics.pt`                      | Pretrained auxiliary prediction model                                   |
| `auxiliary_metrics.input_scaler.pkl`        | Input scaler for the auxiliary model                                    |
| `AlphaCD2_predict.sh`                       | Example script for running AlphaCD2                                     |
| `AlphaCD2_generation_data_preparing.sh`     | Example script for preparing ProGen2 fine-tuning data                   |
| `AlphaCD2_generation_Progen2_finetuning.sh` | Example script for ProGen2 fine-tuning                                  |
| `environment.yml`                           | Conda environment                                                       |
| `test.txt`                                  | Demo dataset containing 152 protein sequences                           |

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/BioMingdi/AlphaCD2.git
cd AlphaCD2
```

The pretrained AlphaCD2 BiLSTM models and auxiliary prediction model required for inference are included in the repository.

---

## 2. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate alphacd2
```

The released environment uses:

```text
Python 3.10
PyTorch 2.8.0
ESM 3.2.1.post1
NumPy 2.2.6
pandas 2.3.2
scikit-learn
```

AlphaCD2 has been developed and tested on Linux.

---

# ESM-C checkpoint

AlphaCD2 uses the **ESM-C 600M** protein language model to generate sequence representations.

The ESM-C checkpoint itself is not distributed in this repository and should be obtained separately from the official ESM release:

https://github.com/evolutionaryscale/esm

The released AlphaCD2 workflow uses a local checkpoint such as:

```text
esmc_600m_2024_12_v0.pth
```

Specify its location using:

```bash
--esm-checkpoint /path/to/esmc_600m_2024_12_v0.pth
```

or set the path in:

```text
AlphaCD2_predict.sh
```

---

# Pretrained AlphaCD2 models

The released on-target predictor consists of three independently initialized full-data BiLSTM models:

```text
seed_42
seed_43
seed_44
```

The corresponding checkpoint and scaler paths are defined in:

```text
bilstm_manifest.json
```

For each protein sequence, AlphaCD2 obtains predictions from all three models. The final:

```text
AlphaCD2_ontarget
```

score is the **mean prediction across the three released models**.

No additional AlphaCD2 checkpoint download is required.

---

# Hardware

AlphaCD2 supports both **GPU and CPU inference**.

A CUDA-compatible NVIDIA GPU is strongly recommended for ESM-C 600M embedding generation, particularly for large sequence sets. CPU inference is supported but can be substantially slower.

Devices can be selected explicitly using:

```bash
--esm-device cuda:0
--predict-device cuda:0
```

Other examples include:

```text
cuda
cuda:0
cuda:1
cpu
```

If CUDA is requested but is unavailable, AlphaCD2 automatically falls back to CPU.

---

# Quick start

## 1. Configure the ESM-C checkpoint

Edit:

```text
AlphaCD2_predict.sh
```

and set:

```bash
ESM_CHECKPOINT="/path/to/esmc_600m_2024_12_v0.pth"
```

The default demo configuration uses:

```bash
--esm-device cuda:0
--predict-device cuda:0
```

Change these options if a different GPU or CPU inference is required.

---

## 2. Run the demo

```bash
bash AlphaCD2_predict.sh
```

The demo analyzes:

```text
test.txt
```

which contains **152 protein sequences**.

In our test environment, th
