# Causality Inspired Brain-Visual Contrastive Learning for Zero-Shot Visual Decoding (CI-BVCL)

## Table of Contents
- **[Introduction](#introduction)**
- **[Repository Structure](#repository-structure)**
- **[Environment Setup](#environment-setup)**
- **[Data Preparation](#data-preparation)**
- **[Running Experiments](#running-experiments)**

## Introduction

This repository provides the official implementation of the paper:

> **Causality Inspired Brain-Visual Contrastive Learning for Zero-Shot Visual Decoding**

We propose a causality-inspired brain–visual contrastive learning framework, **CI-BVCL**, to align brain signals (EEG/MEG) with visual representations and enable **zero-shot visual decoding**. The implementation supports experiments on both **THINGS-EEG** and **THINGS-MEG** datasets, using image features extracted by a CLIP vision encoder.

This codebase is a minimal and clean version tailored for **CI-BVCL**, containing only the components required to reproduce the main experiments of the paper.

## Repository Structure

The main structure of this CI-BVCL repository is as follows:

```text
CI-BVCL/                    # Root directory
├── README.md               # Project documentation (this file)
├── base/                   # Core implementation files
│   ├── data_eeg.py         # EEG data loading (THINGS-EEG)
│   ├── data_meg.py         # MEG data loading (THINGS-MEG)
│   ├── eeg_backbone.py     # Brain encoder: EEG/MEG backbone and project layer
│   └── utils.py            # Utilities: update_config, ClipLoss, instantiate_from_config, get_device, etc.
├── configs/                # Configuration files for CI-BVCL
│   ├── eeg/
│   │   └── cibvcl.yaml     # CI-BVCL configuration for THINGS-EEG
│   └── meg/
│       └── cibvcl.yaml     # (Optional) CI-BVCL configuration for THINGS-MEG
├── data/                   # Directory for datasets (not included)
│   ├── things-eeg/         # THINGS-EEG dataset and metadata
│   └── things-meg/         # THINGS-MEG dataset and metadata
├── preprocess/             # Data preprocessing scripts
│   └── ...                 # Convert raw EEG/MEG to training-ready tensors
├── main_CIBVCL.py            # Main script: CI-BVCL PLModel and training / evaluation loop
└── requirements.txt        # Python dependencies
```

> **Note:** Raw data are **not** included in this repository. Please download THINGS-EEG and THINGS-MEG from their official sources and follow the instructions below to prepare them.

## Environment Setup

- **Python**: 3.9+
- **CUDA**: 11.x / 12.x (depending on your GPU and PyTorch build)
- **PyTorch**: version compatible with your CUDA
- Other dependencies are listed in `requirements.txt`.

### 1. Create Conda environment

```bash
conda create -n cibvcl python=3.9
conda activate cibvcl
```

### 2. Install PyTorch

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

## Data Preparation

We conduct experiments on **THINGS-EEG** and **THINGS-MEG** datasets, together with the **THINGS** image collection.

- [A large and rich EEG dataset for modeling human visual object recognition](https://www.sciencedirect.com/science/article/pii/S1053811922008758) [THINGS-EEG]
- [THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior](https://pubmed.ncbi.nlm.nih.gov/36847339/) [THINGS-MEG]

### Directory layout

After downloading and preprocessing, we get the following structure:

```text
data/
├── things/                 # THINGS images and metadata
│   └── THINGS
│       ├── Images/         # Original images
│       └── Metadata/       # Category / stimulus meta information
├── things-eeg/
│   ├── Image_set/          # THINGS images
│   ├── Image_set_Resize/   # resized THINGS images
|   ├── Image_feature       # Pre-extracted image features
│   ├── Raw_data/           # raw EEG data
│   ├── Preprocessed_data/       # Preprocessed EEG tensors used for training
└── things-meg/
│   ├── Image_set/          # THINGS images
│   ├── Image_set_Resize/   # resized THINGS images
|   ├── Image_feature       # Pre-extracted image features
|   ├── Raw_data/           # raw MEG data
|   ├── Preprocessed_data/       # Preprocessed MEG tensors used for training
```

### Preprocessing

We provide Python scripts under the `preprocess/` directory to convert raw THINGS-EEG/MEG recordings and images into training-ready tensors:

- **`process_eeg_whiten.py`**
  - Preprocess THINGS-EEG data (filtering, epoching, whitening, normalization, etc.).
  - Convert raw EEG into preprocessed tensors saved under `data/things-eeg/Preprocessed_data/`.
  - Example usage:
    ```bash
    python preprocess/process_eeg_whiten.py \
      --dataset things-eeg \
      --out_dir data/things-eeg/Preprocessed_data
    ```
- **`process_meg.py`**
  - Preprocess THINGS-MEG data in a similar manner (cleaning, epoching, normalization).
  - Save processed tensors under `data/things-meg/Preprocessed_data/`.
  - Example usage:
    ```bash
    python preprocess/process_meg.py \
      --dataset things-meg \
      --out_dir data/things-meg/Preprocessed_data
    ```
- **`process_resize.py`**
  - Resize THINGS images to the resolution expected by the CLIP vision encoder.
  - Support both EEG and MEG pipelines via the `--type` argument.
  - Example usage:
    ```bash
    python preprocess/process_resize.py --type eeg
    python preprocess/process_resize.py --type meg
    ```
In addition, use `open_clip_torch` (or your own scripts) to extract image features for the THINGS images, and store them under the `Image_features/` subfolders.

Ensure that the paths in `configs/eeg/cibvcl.yaml` and `configs/meg/cibvcl.yaml` correctly point to your preprocessed data and image features.


### 1. EEG (THINGS-EEG)
Example command:
```bash
python main_CIBVCL.py \
  --config configs/eeg/cibvcl.yaml \
  --dataset eeg \
  --subjects sub-08 \
  --exp_setting intra-subject \
  --brain_backbone EEGProjectLayer \
  --vision_backbone RN50 \
  --epoch 50 \
  --lr 1e-4
```

### 2. MEG (THINGS-MEG)
Example command:
```bash
python main_CIBVCL.py \
  --config configs/meg/cibvcl.yaml \
  --dataset meg \
  --subjects sub-01 \
  --exp_setting intra-subject \
  --brain_backbone EEGProjectLayer \
  --vision_backbone RN50
```
