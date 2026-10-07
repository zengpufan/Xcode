# Multilingual Wikipedia Entity Clustering

A multilingual Wikipedia entity clustering analysis project — performing Named Entity Recognition (NER), text embedding, GPU-accelerated clustering, and quality-optimized selection on cross-lingual Wikipedia data to support cross-lingual semantic alignment and cultural analysis research.

## Overview

This project implements a complete **multilingual Wikipedia text semantic mining pipeline**. It extracts text from 6 language editions of the Wikimedia Wikipedia dataset, filters high-quality articles via spaCy NER, generates cross-lingual embeddings using Sentence-BERT, performs clustering with cuML GPU-accelerated KMeans, optimizes cluster quality via density and semantic entropy, and produces academic-grade visualizations.

**Supported Languages (6)**: `en`, `zh`, `de`, `fr`, `es`, `ja`

## Pipeline

```
Wikipedia Multilingual Data (HuggingFace datasets)
              │
              ▼
    ┌─────────────────┐
    │   [1] NER.py    │  ← spaCy multilingual NER models
    │  negative filter │     (nlp.pipe batched + multiprocessing)
    └────────┬────────┘
             │  {lang}/{lang}_selected_data.jsonl
             ▼
    ┌──────────────────────┐
    │ [2] step1_single_lang │  ← Sentence-BERT + KMeans
    │   _clustering.py      │     + Local Dispersion (Eq. 1)
    │  embed + cluster      │     + Semantic Entropy (Eq. 2-3)
    │  + density/entropy    │     → top-20% by entropy = C_high
    │  quality selection    │
    └────────┬─────────────┘
             │  {lang}_high_quality.jsonl   (sample_high_quality_data.py is a thin wrapper here)
             ▼
    ┌──────────────────────┐
    │ [3] step2_multi_lang  │  ← Sentence-BERT + KMeans(k=1000)
    │   _clustering.py     │     + linguistic dominance (Eq. 4)
    │  Cross-lingual cluster│     → τ=5, θ=0.8
    │  + CP selection       │
    └────────┬─────────────┘
             │  culture_points.json  (Culture Points)
             ▼
    ┌──────────────────────┐
    │ [4] visualize / draw  │  ← PCA / t-SNE dimensionality reduction
    │   Charts & plots      │     Academic charts (pie/radar/bar)
    └──────────────────────┘
```

> Shared configuration lives in `config.py` (paths, model, thresholds);
> shared helpers (embedder singleton, JSONL I/O, geometric metrics, cuML→sklearn
> KMeans fallback) live in `common.py`. Fill in the `*_OUTPUT_DIR` / cache paths
> in `config.py` before running.

## Scripts

### Configuration & shared helpers

| Script | Description |
|--------|-------------|
| `config.py` | All tunable knobs (paths, language list, sampling sizes, spaCy models, embedding model, KNN k, cluster counts, τ=5, θ=0.8, top-20%). Fill in `*_OUTPUT_DIR` here. |
| `common.py` | Embedder singleton, `format_input`/`first_paragraph`, JSONL I/O, `local_dispersion` (Euclidean, Eq. 1), `entropy_of_similarity` (Eq. 2-3), and a cuML→sklearn `get_kmeans` fallback. |

### Pipeline

| Script | Stage | Description |
|--------|-------|-------------|
| `NER.py` | Stage 0 / NER | Multilingual NER negative filter via spaCy `nlp.pipe` (batched) + multiprocessing chunks. Drops functional-category entries and keeps title-coherent articles. Outputs `{lang}/{lang}_selected_data.jsonl` and `{lang}_filtered_data.jsonl` under `NER_OUTPUT_DIR`. |
| `step1_single_lang_clustering.py` | Stage 1 | Encodes `<title> T <text> P1`, KMeans-clusters per language, then applies Local Dispersion (keep δ_m ≤ median) + Semantic Entropy (top-20% by H(D)) **in one pass**, emitting `C_high` as `{lang}_high_quality.jsonl` (plus `{lang}_clusters.json` for inspection). |
| `sample_high_quality_data.py` | Stage 1 (wrapper) | Thin backwards-compatible entry point that runs `step1`'s merged pipeline (the density+entropy logic it used to own now lives in step1 to avoid recomputing embeddings). |
| `step2_multi_lang_clustering.py` | Stage 2 | Projects all `C_high` candidates into the shared multilingual space, KMeans(k=1000), keeps clusters with size ≥ τ=5 and linguistic dominance γ > θ=0.8 (Eq. 4). Outputs `culture_points.json` (the Culture Points) under `MULTI_LANG_OUTPUT_DIR`. |

### Visualization & Analysis

| Script | Description |
|--------|-------------|
| `raw_wiki_data_embedding.py` | Samples raw Wikipedia data, computes embeddings, and visualizes with PCA + t-SNE |
| `visualize_multilang_embeddings.py` | Samples from `culture_points.json` and generates PCA / t-SNE scatter plots of multilingual embeddings |
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

> Edit `scripts/config.py` first: set `DATASET_CACHE_DIR`, `NER_OUTPUT_DIR`,
> `SINGLE_LANG_OUTPUT_DIR`, and `MULTI_LANG_OUTPUT_DIR` to absolute paths. All
> algorithmic defaults (k=5, τ=5, θ=0.8, top-20%, k=1000) already match the paper.

```bash
cd scripts

# 1. NER negative filtering   → NER_OUTPUT_DIR/{lang}/{lang}_selected_data.jsonl
python NER.py

# 2. Stage 1: embed + cluster + density/entropy → C_high per language
python step1_single_lang_clustering.py
#   (or `python sample_high_quality_data.py` — same merged pipeline)

# 3. Stage 2: cross-lingual CP selection → MULTI_LANG_OUTPUT_DIR/culture_points.json
python step2_multi_lang_clustering.py
```

To process a subset of languages, edit `LANG_LIST` in `config.py`. To run on CPU
without cuML, nothing changes — `common.get_kmeans` falls back to sklearn
KMeans automatically. To cap the per-language Wikipedia sample, adjust
`LANG_SAMPLE_SIZE` (defaults: 3M for English, 1M for the rest, per Appendix C).

## Output Directory Structure

```
{NER_OUTPUT_DIR}/{lang}/              # NER-filtered results (jsonl per chunk + merged)
{SINGLE_LANG_OUTPUT_DIR}/             # Stage 1: {lang}_high_quality.jsonl + {lang}_clusters.json
{MULTI_LANG_OUTPUT_DIR}/              # Stage 2: culture_points.json (Culture Points)
wiki_embedding_results/               # Wikipedia embedding PCA/t-SNE plots
combined_embedding_results/           # Dual-source comparison results
```

## Key Highlights

- **Fully GPU-accelerated pipeline**: cuML KMeans + CUDA embeddings + spaCy GPU, significantly boosting large-scale processing efficiency
- **Multiprocessing / multithreading parallelism**: NER chunked processing and multi-language parallel sampling maximize computational resource utilization
- **Semantic entropy as a quality metric**: Innovatively uses semantic entropy to measure intra-cluster semantic consistency for quality optimization
- **Dual-source comparison visualization**: Compares raw Wikipedia distributions against cluster-filtered distributions, intuitively revealing data filtering effects
- **Publication-ready chart output**: Generates pie charts, radar charts, bar charts, line charts, and multilingual scatter plots suitable for academic papers

