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

AlphaCD2 encodes protein sequences with ESM-C 600M and uses the resulting representations for activity and auxiliary-property prediction. The activity score is averaged across three BiLSTM models trained with different random seeds.

For protein-design applications, ProGen2-generated candidates can be scored with AlphaCD2 and prioritized for experimental validation.

---

## Repository contents

### Main files

| File                                        | Description                                                             |
| ------------------------------------------- | ----------------------------------------------------------------------- |
| `AlphaCD2_predict.py`                       | Unified AlphaCD2 prediction pipeline                                    |
| `AlphaCD2_ontarget.py`                      | ESM-C embedding generation and BiLSTM on-target prediction              |
| `embedding.py` | Generate ESM-C embeddings from a labelled training dataset |
| `AlphaCD2_activity_train.py` | Select training duration and train the final BiLSTM activity models |
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

In our test environment, the complete prediction workflow for the 152-sequence demo finishes in **less than 1 minute using a CUDA-enabled GPU**.

The runtime may vary depending on GPU model, hardware configuration, software environment, and whether ESM-C embeddings have already been cached.

---

## 3. Demo output

The prediction results are written to:

```text
test_result.tsv
```

A run summary is additionally written to:

```text
test_result.tsv.summary.json
```

ESM-C embeddings are cached as:

```text
test_result.embeddings.pkl
```

and can be reused during subsequent runs.

---

# Input format

AlphaCD2 accepts a two-column text file containing a sequence identifier and an amino-acid sequence:

```text
sequence_name    protein_sequence
```

Example:

```text
NeoCD001    MAAGPAPEARPLMDKQTFLDNFSHL...
NeoCD002    MATAGPSPEARPLMDEQTFLDNFSH...
NeoCD003    MEAGPAPEARPLMDEQTFLDNFSHL...
```

Tab-separated and whitespace-separated files are supported.

Sequence names must be unique.

If the input contains a header:

```text
name    sequence
```

add:

```bash
--has-header
```

to the prediction command.

---

# Running AlphaCD2 on your own sequences

The complete prediction command is:

```bash
python AlphaCD2_predict.py \
    --input-txt my_sequences.txt \
    --output-tsv my_predictions.tsv \
    --ontarget-script AlphaCD2_ontarget.py \
    --manifest bilstm_manifest.json \
    --esm-checkpoint /path/to/esmc_600m_2024_12_v0.pth \
    --specificity-checkpoint auxiliary_metrics.pt \
    --esm-device cuda:0 \
    --predict-device cuda:0
```

For CPU inference:

```bash
python AlphaCD2_predict.py \
    --input-txt my_sequences.txt \
    --output-tsv my_predictions.tsv \
    --ontarget-script AlphaCD2_ontarget.py \
    --manifest bilstm_manifest.json \
    --esm-checkpoint /path/to/esmc_600m_2024_12_v0.pth \
    --specificity-checkpoint auxiliary_metrics.pt \
    --esm-device cpu \
    --predict-device cpu
```

For all available options:

```bash
python AlphaCD2_predict.py --help
```

---

# Output

The output TSV contains eight columns:

```text
prediction_rank
name
AlphaCD2_ontarget
Motif
Window_start
Window_end
Window
Offtarget
```

| Column              | Description                                                                            |
| ------------------- | -------------------------------------------------------------------------------------- |
| `prediction_rank`   | Rank according to predicted on-target activity; rank 1 is the highest-scoring sequence |
| `name`              | Input sequence identifier                                                              |
| `AlphaCD2_ontarget` | Predicted on-target activity, averaged across the three released BiLSTM models         |
| `Motif`             | Predicted preferred cytidine motif                                                     |
| `Window_start`      | Predicted start position of the editing window                                         |
| `Window_end`        | Predicted end position of the editing window                                           |
| `Window`            | Compact representation of the predicted editing window, e.g. `4_8`                     |
| `Offtarget`         | Predicted sgRNA-independent off-target activity                                        |

### Motif prediction

Possible motif classes are:

```text
AC
CC
GC
TC
```

### Editing-window prediction

The auxiliary model predicts editing activity across positions **1–14** and reports a contiguous editing window.

For example:

```text
Window_start = 4
Window_end   = 8
Window       = 4_8
```

### Off-target prediction

`Offtarget` represents the predicted **sgRNA-independent off-target activity** of the cytidine deaminase.

It should not be interpreted as a prediction of guide-dependent genomic off-target editing.

---

# ESM-C embedding cache

AlphaCD2 stores generated ESM-C embeddings so they can be reused without recomputing the representations.

By default, the cache file is generated from the output filename.

A custom cache file can be specified using:

```bash
--embedding-cache-pkl my_embeddings.pkl
```

To force regeneration of all embeddings:

```bash
--regenerate-embeddings
```

If a sequence identifier already exists in the cache but its protein sequence has changed, AlphaCD2 stops and requests regeneration of the embeddings.

---

# Device selection

The following device specifications are supported:

```text
cpu
cuda
cuda:0
cuda:1
...
```

For example:

```bash
--esm-device cuda:0 \
--predict-device cuda:0
```

ESM-C embedding generation and downstream AlphaCD2 prediction can also use different GPUs:

```bash
--esm-device cuda:0 \
--predict-device cuda:1
```

The program checks whether the requested CUDA device exists before prediction.

---

# Training the activity predictor

The following steps train the BiLSTM on-target activity predictor from a labelled protein dataset. Run the commands from the repository root, with `alphacd2_seed42_model.py` available alongside `AlphaCD2_activity_train.py`.

## 1. Prepare the training data

Create a **tab-separated file without a header** containing three columns:

| Column | Content |
| --- | --- |
| 1 | Unique protein identifier |
| 2 | Complete amino-acid sequence |
| 3 | Experimentally measured editing efficiency |

Use editing efficiencies on the same scale as the original training labels. For fractional labels, 25% editing is stored as `0.25`. Neither script automatically converts percentage values to fractions. Keep the scale consistent throughout training and prediction.

Use unique identifiers and check for duplicate sequences before training. The two-column inference demo `test.txt` does not contain the activity labels required for this step.

## 2. Generate ESM-C embeddings

The standalone `embedding.py` script uses ESM-C 600M and reads its input and output paths from the final block of the script:

```python
input_file = "train_noCR.txt"
output_file = "apobec_embeddings_noCR.pkl"
```

Edit these paths if needed. Place `esmc_600m_2024_12_v0.pth` in the repository root or at `data/weights/esmc_600m_2024_12_v0.pth`. If the checkpoint is in the root and the expected weights path does not exist, the script creates a symbolic link to it.

Run:

```bash
python embedding.py
```

This standalone script currently sets its device in `setup_model()`:

```python
device = torch.device("cpu")
```

To use a CUDA GPU, change that line to the appropriate device, for example `torch.device("cuda:0")`. Unlike the unified inference pipeline, this script does not accept `--esm-device` or `--esm-checkpoint` arguments.

The output pickle maps each protein identifier to its sequence, embedding tensor and experimental efficiency. The script saves the returned ESM-C embedding tensor directly, without mean pooling. It reports failed sequences and skips them; check the final counts before training.

## 3. Train the final BiLSTM models

```bash
python AlphaCD2_activity_train.py \
    --data-pickle apobec_embeddings_noCR.pkl \
    --out-dir activity_training \
    --device cuda:0 \
    --final-seeds 42 43 44
```

For CPU training, use `--device cpu`.

By default, the script uses random five-fold validation to select the training duration. It records the best epoch in each fold, takes their median, and trains three new models on the complete dataset with seeds 42, 43 and 44.

The five-fold stage is used for **epoch selection**, not independent performance estimation. Its best validation scores should not be reported as nested out-of-fold test performance. This script does not implement the separate nested cross-validation evaluation used to assess generalization.

Before model fitting, each saved embedding is flattened and truncated to 1,152 elements, or zero-padded if shorter. Features are standardized using the corresponding training data. This preprocessing does not perform mean pooling over residues.

Default settings:

| Parameter | Default |
| --- | --- |
| Input dimension | 1,152 |
| Epoch-selection folds | 5 |
| Maximum epoch-selection epochs | 300 |
| Early-stopping patience | 30 |
| Training batch size | 32 |
| Optimizer | AdamW |
| Maximum learning rate | 5e-4 |
| Weight decay | 1e-5 |
| Learning-rate schedule | OneCycleLR |
| Dropout | 0.30 |
| BiLSTM hidden dimension | 256 |
| BiLSTM layers | 2 |
| Final training seeds | 42, 43, 44 |

To use a predetermined training duration and skip epoch selection, pass a positive `--final-epochs` value. For example, `--final-epochs 100` trains each final model for 100 epochs; this is an example, not a reported training duration for the released checkpoints.

For all options:

```bash
python AlphaCD2_activity_train.py --help
```

## 4. Training outputs

The output directory contains:

| Path | Content |
| --- | --- |
| `dataset_summary.json` | Record counts, embedding shapes, label summary and skipped records |
| `epoch_selection_fold*.tsv` | Per-fold training and validation histories, when epoch selection is run |
| `epoch_selection_fold_assignments.tsv` | Dataset row indices and their validation-fold assignments |
| `epoch_selection.json` | Fold-specific best epochs and selected final duration |
| `seed_42/`, `seed_43/`, `seed_44/` | Final checkpoint, scaler, training history and training records for each seed |
| `final_bilstm_manifest.json` | Paths and metadata for the newly trained activity models |

Each seed directory contains `bilstm_checkpoint.pt`, `scaler.pkl`, `training_history.tsv` and `training_records.tsv`. Keep the manifest and seed directories together because model paths in the manifest are relative to its directory.

The newly generated `final_bilstm_manifest.json` is separate from the repository's released `bilstm_manifest.json`. These training steps train only the activity predictor; they do not retrain the motif, editing-window or off-target model.

---

# ProGen2 workflow

AlphaCD2 can be combined with **ProGen2** for language-model-guided generation of cytidine deaminase variants.

ProGen2 is an external dependency and is not included in this repository.

The released repository provides scripts for:

1. preparing a cytidine deaminase sequence dataset for ProGen2; and
2. fine-tuning ProGen2 on the resulting training dataset.

---

## 1. Prepare ProGen2 data

Edit:

```text
AlphaCD2_generation_data_preparing.sh
```

and configure:

```bash
PROGEN2_DIR="/path/to/ProGen2-finetuning"
INPUT_FASTA="/path/to/CD_sequences.fasta"
TRAIN_FILE="./CD_train.json"
TEST_FILE="./CD_test.json"
```

Then run:

```bash
bash AlphaCD2_generation_data_preparing.sh
```

The released example uses:

```text
training fraction = 0.8
bidirectional sequence processing
random seed = 2025
```

---

## 2. Fine-tune ProGen2

Edit:

```text
AlphaCD2_generation_Progen2_finetuning.sh
```

and configure the ProGen2 installation and training/test files.

Then run:

```bash
bash AlphaCD2_generation_Progen2_finetuning.sh
```

The released example fine-tunes:

```text
hugohrban/progen2-base
```

using:

```text
epochs               = 5
batch size            = 2
gradient accumulation = 4
learning rate         = 1e-4
learning-rate decay   = cosine
warmup steps          = 200
random seed           = 2025
```

After fine-tuning, generated candidate sequences can be evaluated and ranked using `AlphaCD2_predict.py`.

---

# Scope and interpretation

AlphaCD2 is intended as a **computational prioritization tool** for cytidine deaminase variants.

Predicted scores should not be considered substitutes for experimental validation.

Prediction performance may decrease as candidate proteins become increasingly distant from the sequence space represented in the training dataset. Predictions for highly divergent sequences should therefore be interpreted cautiously.

The motif, editing-window, and sgRNA-independent off-target predictors are intended to assist candidate prioritization and experimental design rather than replace direct experimental characterization.

---

# Code availability

This repository provides AlphaCD2 inference code, pretrained models and final activity-model training code, including:

* ESM-C representation generation for inference and labelled training data;
* final BiLSTM activity-model training with epoch selection and multiple random seeds;
* BiLSTM-based on-target activity prediction;
* motif-preference prediction;
* editing-window prediction;
* sgRNA-independent off-target prediction;
* pretrained AlphaCD2 checkpoints and preprocessing scalers; and
* scripts for ProGen2 data preparation and fine-tuning.

The activity-training workflow is provided in `embedding.py` and `AlphaCD2_activity_train.py`. These scripts cover embedding extraction and final activity-model fitting. They do not include auxiliary-model training or the complete nested cross-validation benchmarking workflow.

The released pretrained model checkpoints are sufficient for applying AlphaCD2 to new protein sequences.

---

# Citation

If you use AlphaCD2 in your research, please cite the associated AlphaCD2 manuscript.

Full citation information will be added upon publication.

---

# License

This project is distributed under the MIT License.

See:

```text
LICENSE
```

for details.

---

# Contact

For questions regarding AlphaCD2, please contact:

```text
mingdiwu0228@163.com
```
