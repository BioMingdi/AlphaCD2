# AlphaCD2

AlphaCD2 is a structure-independent computational framework for designing and selecting cytidine deaminases. AlphaCD2 couples an ESM-C-based activity predictor with ProGen2-based sequence generation to create and select novel cytidine deaminases, termed NeoCDs, without requiring structural templates, structure prediction, or structural alignment.

This repository provides standalone source code, shell scripts, a small demo dataset, and the expected demo output for running the AlphaCD2 prediction and ProGen2 generation workflow.

---

## Repository contents

```text
AlphaCD2/
├── AlphaCD2_predicting.py
├── AlphaCD2_predicting.sh
├── AlphaCD2_generation_data_preparing.sh
├── AlphaCD2_generation_Progen2_finetuning.sh
├── environment.yml
├── test.txt
├── test_efficiency_pre.txt
├── README.md
└── LICENSE
```

### File description

| File | Description |
|---|---|
| `AlphaCD2_predicting.py` | Source code for AlphaCD2 activity prediction. |
| `AlphaCD2_predicting.sh` | Shell script for running the AlphaCD2 prediction demo. |
| `AlphaCD2_generation_data_preparing.sh` | Shell script for preparing ProGen2 fine-tuning data. |
| `AlphaCD2_generation_Progen2_finetuning.sh` | Shell script for ProGen2 fine-tuning and sequence generation. |
| `environment.yml` | Conda environment file containing the required software dependencies. |
| `test.txt` | Small demo input dataset. The first column is the sequence name and the second column is the protein sequence. |
| `test_efficiency_pre.txt` | Expected output file for the demo prediction. |
| `LICENSE` | License information. |
| `README.md` | Documentation for installation, demo running, and software usage. |

---

# Compiled standalone software and/or source code

This repository contains standalone source code and shell scripts for running AlphaCD2.

The main source code file is:

```text
AlphaCD2_predicting.py
```

The main executable shell scripts are:

```text
AlphaCD2_predicting.sh
AlphaCD2_generation_data_preparing.sh
AlphaCD2_generation_Progen2_finetuning.sh
```

No compilation is required for the AlphaCD2 prediction demo. The software can be run directly after installing the required Python dependencies.

---

# Small demo dataset

A small demo dataset is provided as:

```text
test.txt
```

The file contains two columns:

```text
sequence_name    protein_sequence
```

The first column is the sequence identifier, and the second column is the amino acid sequence.

Example format:

```text
seq_1    MXXX...
seq_2    MXXX...
seq_3    MXXX...
```

The expected demo output is provided as:

```text
test_efficiency_pre.txt
```

This file contains the predicted cytidine deaminase activity scores for the demo sequences in `test.txt`.

---

# 1. System requirements

## Operating system

AlphaCD2 has been tested on Linux operating systems.

Tested systems include:

```text
Ubuntu 20.04 LTS
Ubuntu 22.04 LTS
CentOS 7
```

The demo prediction is expected to run on most Linux systems with a working Conda installation.

## Software dependencies

The required software dependencies are listed in:

```text
environment.yml
```

Main dependencies include:

```text
Python
PyTorch
NumPy
pandas
scikit-learn
SciPy
Biopython
tqdm
joblib
transformers
tokenizers
```

The exact package versions used for the released environment are specified in `environment.yml`.

After installation, the versions can be checked using:

```bash
conda activate alphacd2

python --version

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"

python -c "import numpy, pandas, sklearn, scipy; print('NumPy:', numpy.__version__); print('pandas:', pandas.__version__); print('scikit-learn:', sklearn.__version__); print('SciPy:', scipy.__version__)"
```

## Hardware requirements

For the small demo prediction using `test.txt`:

```text
A standard desktop or laptop computer is sufficient.
GPU is not required.
Recommended memory: 8 GB RAM or higher.
```

For large-scale ProGen2 fine-tuning and sequence generation:

```text
An NVIDIA GPU is recommended.
CUDA-compatible GPU memory is required for large ProGen2 models.
Recommended system memory: 32 GB RAM or higher.
```

No non-standard hardware is required for running the AlphaCD2 prediction demo.

---

# 2. Installation guide

## Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AlphaCD2.git
cd AlphaCD2
```

If the repository has already been downloaded, enter the repository directory directly:

```bash
cd AlphaCD2
```

## Step 2: Create the Conda environment

```bash
conda env create -f environment.yml
```

## Step 3: Activate the environment

```bash
conda activate alphacd2
```

If the environment name in `environment.yml` is different, please activate the corresponding environment name.

## Step 4: Check the installation

```bash
python --version
python -c "import torch; print(torch.__version__)"
```

## Typical installation time

On a normal desktop computer with a stable internet connection, installation usually takes:

```text
5–15 minutes for standard Python dependencies.
15–30 minutes if PyTorch or CUDA packages need to be downloaded.
```

Installation time may vary depending on internet speed and whether GPU-enabled PyTorch is installed.

No compilation step is required for the AlphaCD2 prediction demo.

---

# 3. Demo

The demo uses the provided input file:

```text
test.txt
```

The expected output is provided as:

```text
test_efficiency_pre.txt
```

## Run the demo

After activating the Conda environment, run:

```bash
bash AlphaCD2_predicting.sh
```

This script runs AlphaCD2 activity prediction on the demo dataset.

Alternatively, the Python script can be run directly:

```bash
python AlphaCD2_predicting.py
```

If your local version of `AlphaCD2_predicting.py` requires explicit input and output arguments, run:

```bash
python AlphaCD2_predicting.py \
    --input test.txt \
    --output test_efficiency_pre.txt
```

## Demo input

The demo input file is:

```text
test.txt
```

Input format:

```text
sequence_name    protein_sequence
```

Example:

```text
candidate_001    MXXX...
candidate_002    MXXX...
candidate_003    MXXX...
```

## Expected output

The expected demo output file is:

```text
test_efficiency_pre.txt
```

The output contains predicted cytidine deaminase activity scores for the sequences in `test.txt`.

Expected output format:

```text
sequence_name    predicted_activity
```

or, depending on the script version:

```text
sequence_name    protein_sequence    predicted_activity
```

Minor numerical differences may occur across different operating systems, PyTorch versions, CUDA versions, or CPU/GPU settings.

## Expected run time for demo

On a normal desktop computer:

```text
Expected run time: less than 1 minute.
```

On a GPU workstation:

```text
Expected run time: several seconds to less than 1 minute.
```

The demo is intended to verify that the software and environment are correctly installed. It is not intended to reproduce the full-scale sequence generation and screening experiments from the manuscript.

---

# 4. Instructions for use

AlphaCD2 can be used in two main modes:

```text
1. Predicting the activity of user-provided cytidine deaminase sequences.
2. Preparing, fine-tuning, and using ProGen2 to generate new candidate cytidine deaminases.
```

---

## 4.1 Activity prediction for user-provided sequences

Prepare an input text file using the same format as `test.txt`.

Input file format:

```text
sequence_name    protein_sequence
```

Example:

```text
my_CD_001    MXXX...
my_CD_002    MXXX...
my_CD_003    MXXX...
```

Then run:

```bash
python AlphaCD2_predicting.py \
    --input my_sequences.txt \
    --output my_predictions.txt
```

If using the provided shell script, modify the input and output filenames in:

```text
AlphaCD2_predicting.sh
```

Then run:

```bash
bash AlphaCD2_predicting.sh
```

The output file will contain predicted activity scores for each input sequence.

---

## 4.2 ProGen2 data preparation

To prepare sequence data for ProGen2 fine-tuning, run:

```bash
bash AlphaCD2_generation_data_preparing.sh
```

This script prepares ProGen2-compatible training and testing files from the input cytidine deaminase sequence dataset.

The typical workflow is:

```text
Input cytidine deaminase sequences
        ↓
ProGen2-compatible training file
        ↓
ProGen2-compatible testing file
```

Before running the script on your own data, edit the input filename, output training filename, and output testing filename in:

```text
AlphaCD2_generation_data_preparing.sh
```

---

## 4.3 ProGen2 fine-tuning and sequence generation

To fine-tune ProGen2 and generate new cytidine deaminase sequences, run:

```bash
bash AlphaCD2_generation_Progen2_finetuning.sh
```

This script performs ProGen2 fine-tuning and sequence generation using the specified model, training file, testing file, and sampling parameters.

Main parameters include:

```text
model
train_file
test_file
device
epochs
batch_size
accumulation_steps
learning_rate
temperature
top_k
maximum sequence length
prompt
random seed
```

Before running the script on your own data, edit the file paths and parameters in:

```text
AlphaCD2_generation_Progen2_finetuning.sh
```

For large-scale generation, a CUDA-compatible NVIDIA GPU is recommended.

---

## 4.4 Running AlphaCD2 on your own data

To run AlphaCD2 on your own cytidine deaminase sequences:

### Step 1: Prepare input file

Create a text file with two columns:

```text
sequence_name    protein_sequence
```

For example:

```text
my_CD_001    MXXX...
my_CD_002    MXXX...
my_CD_003    MXXX...
```

Save this file as:

```text
my_sequences.txt
```

### Step 2: Run activity prediction

```bash
python AlphaCD2_predicting.py \
    --input my_sequences.txt \
    --output my_predictions.txt
```

Alternatively, modify `AlphaCD2_predicting.sh` and run:

```bash
bash AlphaCD2_predicting.sh
```

### Step 3: Check output

The output file contains predicted activity scores for each input sequence:

```text
my_predictions.txt
```

Sequences with higher predicted activity can be selected for downstream experimental validation.

---

# Notes

AlphaCD2 does not require:

```text
Experimentally solved protein structures
Homology models
AlphaFold-predicted structures
Structural alignment
Structural clustering
```

The workflow is based on protein sequence representation, activity prediction, ProGen2-based generation, and candidate selection.

---

# Citation

If you use AlphaCD2, please cite:

```text
Wu M. et al. One-shot generation of cytidine deaminases using large language models without structural information.
```

---

# License

This project is distributed under the license provided in:

```text
LICENSE
```

---

# Contact

For questions, please contact:

```text
Mingdi Wu
```
