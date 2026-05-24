# Multilingual Wikipedia Entity Clustering

A multilingual Wikipedia entity clustering analysis project — performing Named Entity Recognition (NER), text embedding, GPU-accelerated clustering, and quality-optimized selection on cross-lingual Wikipedia data to support cross-lingual semantic alignment and cultural analysis research.

## Overview

This project implements a complete **multilingual Wikipedia text semantic mining pipeline**. It extracts text from 9 language editions of the Wikimedia Wikipedia dataset, filters high-quality articles via spaCy NER, generates cross-lingual embeddings using Sentence-BERT, performs clustering with cuML GPU-accelerated KMeans, optimizes cluster quality via density and semantic entropy, and produces academic-grade visualizations.

**Supported Languages (9)**: `en`, `zh`, `de`, `fr`, `es`, `it`, `nl`, `pt`, `ja`

## Pipeline

```
Wikipedia Multilingual Data (HuggingFace datasets)
              │
              ▼
    ┌─────────────────┐
    │   [1] NER.py    │  ← spaCy multilingual NER models
    │  Entity recogn.  │
    │  + filtering     │
    └────────┬────────┘
             │  {lang}_selected_data.json
             ▼
    ┌──────────────────┐
    │ [2] single_lang  │  ← Sentence-BERT + cuML KMeans
    │   _cluster.py    │
    │  Per-lang embed   │
    │  + clustering     │
    └────────┬─────────┘
             │  {lang}_clusters.json
             ▼
    ┌─────────────────┐
    │  [3] sample.py  │  ← KNN density + semantic entropy
    │  Density-based   │
    │  quality selection│
    └────────┬────────┘
             │  top_clusters_{lang}.json
             ▼
    ┌──────────────────┐
    │ [4] multi_lang   │  ← Sentence-BERT + cuML KMeans
    │   _cluster.py    │
    │  Cross-lingual    │
    │  clustering       │
    └────────┬─────────┘
             │  single_lang_clusters.json
             ▼
    ┌──────────────────────┐
    │ [5] visualize / draw │  ← PCA / t-SNE dimensionality reduction
    │   Charts & plots     │     Academic charts (pie/radar/bar)
    └──────────────────────┘
```

## Scripts

### 1. Data Processing

| Script | Description |
|--------|-------------|
| `NER.py` | Performs multilingual NER (LOC/ORG/GPE/DATE/FAC etc.) on Wikipedia data using spaCy. Validates entity relevance against article titles and filters invalid articles. Uses multiprocessing for parallel processing. Outputs `{lang}_selected_data.json` and `{lang}_filtered_data.json` under `ner_processed_data/` |
| `X_NER_optimized.py` | Optimized variant using spaCy **transformer models** (`en_core_web_trf` etc.) — higher accuracy but slower |

### 2. Embedding & Clustering

| Script | Description |
|--------|-------------|
| `single_lang_cluster.py` | Generates text embeddings per language using `paraphrase-multilingual-mpnet-base-v2` and runs GPU-accelerated KMeans clustering. Outputs `{lang}_clusters.json` |
| `X_single_lang_cluster_compatible.py` | Compatible variant with automatic fallback from cuML to sklearn KMeans |
| `sample.py` | Computes KNN cosine-similarity density and semantic entropy for each cluster, selects the top 20% by quality. Outputs `top_clusters_{lang}.json` and `rest_clusters_{lang}.json` |
| `multi_lang_cluster.py` | Mixes high-quality cluster texts from all languages for cross-lingual joint clustering (k=1000). Filters single-language-dominant clusters (≥ 80% of one language). Outputs `single_lang_clusters.json` |

### 3. Visualization & Analysis

| Script | Description |
|--------|-------------|
| `raw_wiki_data_embedding.py` | Samples raw Wikipedia data, computes embeddings, and visualizes with PCA + t-SNE |
| `visualize_multilang_embeddings.py` | Samples from `single_lang_clusters.json` and generates PCA / t-SNE scatter plots of multilingual embeddings |
| `two_source_visualization.py` | Dual-source comparison (raw Wikipedia vs. cluster-filtered data), produces 6 comparison plots |
| `draw_cp_human_eval.py` | Grouped bar chart of human evaluation scores |
| `draw_pie_chart.py` | Language distribution pie chart |
| `draw_radar.py` | 5-dimensional radar chart comparing culture-relevant vs. culture-irrelevant points |
| `analyse_theta.py` | Theta hyperparameter analysis line chart (score trends across different models and parameters) |
| `test.py` | Proof-of-concept scatter plot with simulated multi-country cluster data |

## Dependencies

### Core

```bash
pip install torch datasets sentence-transformers spacy scikit-learn scipy matplotlib numpy tqdm
```

### GPU Acceleration (Recommended)

```bash
# cuML (RAPIDS) — GPU-accelerated KMeans
pip install cuml-cu12  # Choose cu11/cu12 based on your CUDA version
```

### spaCy Language Models

```bash
python -m spacy download en_core_web_lg
python -m spacy download zh_core_web_lg
python -m spacy download de_core_news_lg
python -m spacy download fr_core_news_lg
python -m spacy download es_core_news_lg
python -m spacy download it_core_news_lg
python -m spacy download nl_core_news_lg
python -m spacy download pt_core_news_lg
python -m spacy download ja_core_news_lg
# Optimized variant (X_NER_optimized.py) additionally requires:
python -m spacy download en_core_web_trf
python -m spacy download zh_core_web_trf
```

### System Requirements

- Python ≥ 3.10
- CUDA ≥ 12.0 (recommended for GPU acceleration)
- Sufficient disk space (for caching Wikipedia datasets)

## Quick Start

### 1. Data Preprocessing (NER Filtering)

```bash
python NER.py
```

Output directory: `ner_processed_data/`

### 2. Per-Language Clustering

```bash
python single_lang_cluster.py
```

Output directory: `output_single_lang_clusters/`

### 3. Density Sampling & Quality Selection

```bash
python sample.py
```

Output directory: `sample_result/`

### 4. Cross-Lingual Joint Clustering

```bash
python multi_lang_cluster.py
```

Output directory: `multi_lang_output/`

### 5. Visualization

```bash
# Multilingual embedding visualization
python visualize_multilang_embeddings.py

# Dual-source comparison visualization
python two_source_visualization.py

# Raw Wikipedia embedding visualization
python raw_wiki_data_embedding.py
```

## Output Directory Structure

```
ner_processed_data/          # NER-filtered results
output_single_lang_clusters/ # Per-language clustering results
sample_result/               # Top-K density-based sampling results
multi_lang_output/           # Cross-lingual joint clustering results
wiki_embedding_results/      # Wikipedia embedding PCA/t-SNE plots
combined_embedding_results/  # Dual-source comparison results
```

## Key Highlights

- **Fully GPU-accelerated pipeline**: cuML KMeans + CUDA embeddings + spaCy GPU, significantly boosting large-scale processing efficiency
- **Multiprocessing / multithreading parallelism**: NER chunked processing and multi-language parallel sampling maximize computational resource utilization
- **Semantic entropy as a quality metric**: Innovatively uses semantic entropy to measure intra-cluster semantic consistency for quality optimization
- **Dual-source comparison visualization**: Compares raw Wikipedia distributions against cluster-filtered distributions, intuitively revealing data filtering effects
- **Publication-ready chart output**: Generates pie charts, radar charts, bar charts, line charts, and multilingual scatter plots suitable for academic papers

