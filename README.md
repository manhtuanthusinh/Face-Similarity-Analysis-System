# Face Image Analytics

A face recognition pipeline that extracts embeddings from facial images and evaluates verification performance through statistical metrics.

## Overview

This project implements an end-to-end face recognition system leveraging **AdaFace**, a quality-adaptive margin-based face recognition model. The system processes raw face images through face alignment, extracts discriminative embeddings, and performs similarity-based matching with configurable thresholds.

## Key Features

- **Face Detection & Alignment**: MTCNN-based face landmark detection and alignment to 112x112x3 standard input
- **Deep Embedding Extraction**: AdaFace IR-18 backbone with WebFace4M pretrained weights
- **Face Verification**: Cosine similarity-based matching with configurable thresholds
- **Performance Evaluation**: FAR/FRR analysis with EER (Equal Error Rate) computation
- **Adaptive Thresholding**: Per-sample threshold calculation for improved separability
- **Batch Processing**: Efficient handling of large datasets with progress tracking
- **Query Matching**: Match query images against a pre-built embedding database

## Tech Stack

| Component | Technology |
|-----------|------------|
| Data Processing | NumPy, pandas, scikit-learn |
| Visualization | Matplotlib |
| Model | AdaFace IR-18 (WebFace4M) |

## Architecture

```
Input Images → Face Detection (MTCNN) → Face Alignment → Preprocessing
                                                              ↓
                                                    AdaFace IR-18 Backbone
                                                              ↓
                                                    512-dim Embedding Vector
                                                              ↓
                            ┌──────────────────────┬──────────────────────┐
                            ↓                      ↓                      ↓
                     Face Matching           FAR/FRR Analysis       Database Index
                  (Cosine Similarity)       (Threshold Search)      (Query Match)
```

## Project Structure

```
├── config.py              # Configuration (paths, device, model settings)
├── main.py                # Embedding extraction to CSV format
├── main1.py               # Embedding extraction to NumPy format
├── correlation.py          # Similarity matrix & FAR/FRR analysis
├── face_query_match.py    # Query-to-database face matching
├── evaluate_far_frr.py     # Threshold-based FAR/FRR evaluation
├── extract_img_aligned_with_name.py  # Dataset preparation utility
├── core/
│   ├── data_load.py       # Image loading & person name extraction
│   ├── model_load.py      # Model initialization & input preprocessing
│   ├── safe_align_face.py # Robust face alignment wrapper
│   ├── metrics.py         # FAR/FRR computation
│   └── l2_norm_cosine_func.py  # Similarity functions
├── AdaFace/               # AdaFace model implementation (submodule)
├── pretrained/            # Model checkpoints
├── data/                  # Input datasets
└── output/                # Extracted embeddings & results
```

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster processing)

### Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate adaface

# Download pretrained model (IR-18 WebFace4M)
# Place in pretrained/adaface_ir18_webface4m.ckpt
```

### Configuration

Edit `config.py` to customize:

```python
DATASET_PATH = "data/dataface_au/DB_face_au"  # Input dataset
DEVICE = "cuda:0"  # or "cpu"
MODEL_CKPT = "pretrained/adaface_ir18_webface4m.ckpt"
```

## Usage

### 1. Extract Embeddings

```bash
# Extract to NumPy format (recommended)
python main1.py

# Extract to CSV format
python main.py
```

### 2. Analyze Similarity & Metrics

```bash
# Compute correlation matrix and FAR/FRR analysis
python correlation.py
```

### 3. Face Query Matching

```bash
# Match query images against database
python face_query_match.py
```

## Example Output

**Correlation Analysis:**
- EER (Equal Error Rate) and optimal threshold identification
- Intra-class vs Inter-class similarity distribution
- Separability gap analysis per sample

**Query Matching Results:**
- Excel output with person ID, image name, matched identity, and similarity score
- Configurable matching threshold for precision/recall trade-off

## Technical Details

### Model: AdaFace IR-18

AdaFace improves face recognition under varying image quality by:
- Adapting the margin function based on feature norm (proxy for image quality)
- Assigning higher importance to high-quality hard samples
- Achieving state-of-the-art results on IJB-B, IJB-C, IJB-S, and TinyFace

### Embedding Pipeline

1. **Image Loading**: PIL + OpenCV for RGB/BGR handling
2. **Face Detection**: MTCNN detects 5 facial landmarks
3. **Alignment**: Affine transformation to normalized face pose
4. **Preprocessing**: BGR conversion, normalization (mean=0.5, std=0.5)
5. **Inference**: Forward pass through IR-18 backbone
6. **Normalization**: L2 normalization for cosine similarity compatibility

### Evaluation Metrics

- **FAR (False Acceptance Rate)**: Proportion of inter-class pairs incorrectly matched
- **FRR (False Rejection Rate)**: Proportion of intra-class pairs incorrectly rejected
- **EER (Equal Error Rate)**: Threshold where FAR equals FRR
- **Adaptive Threshold**: Per-sample threshold optimizing intra/inter-class separation

## Performance Notes

- CPU inference: ~0.5s per image
- GPU inference (CUDA): ~0.05s per image
- Memory usage: ~2GB for IR-18 model
- Embedding dimension: 512 features per face


