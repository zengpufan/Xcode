"""Central configuration for the C-Mining pipeline.

Fill in the INPUT/OUTPUT paths before running. All algorithmic constants
default to the values reported in the paper (arXiv:2604.15675).
"""
from pathlib import Path

# --- Paths (TODO: fill in before running) ---
DATASET_CACHE_DIR = ""            # HuggingFace datasets cache dir
NER_OUTPUT_DIR = ""               # step0 output: {lang}_selected_data.jsonl
SINGLE_LANG_OUTPUT_DIR = ""       # step1 output: {lang}_high_quality.jsonl
MULTI_LANG_OUTPUT_DIR = ""        # step2 output: culture_points.json

# --- Dataset ---
DATASET_NAME = "wikimedia/wikipedia"
DATASET_DATE = "20231101"
LANG_LIST = ["en", "zh", "de", "fr", "es", "ja"]

# Stratified down-sampling per language (Appendix C).
# Non-English languages are normalized to 1M entries; English is oversampled
# to 3M to act as a dense reference for cross-lingual filtering.
LANG_SAMPLE_SIZE = {
    "en": 3_000_000,
    "default": 1_000_000,
}

# --- spaCy NER (step0) ---
SPACY_MODEL = {
    "en": "en_core_web_lg",
    "zh": "zh_core_web_lg",
    "de": "de_core_news_lg",
    "fr": "fr_core_news_lg",
    "es": "es_core_news_lg",
    "ja": "ja_core_news_lg",
}
NER_VALID_ENTITIES = {
    "en": ["DATE", "FAC", "GPE", "LOC", "ORG", "WORK_OF_ART", "NORP"],
    "zh": ["DATE", "FAC", "GPE", "LOC", "ORG", "WORK_OF_ART", "NORP"],
    "ja": ["DATE", "FAC", "GPE", "LOC", "ORG", "WORK_OF_ART", "NORP"],
    "default": ["LOC", "ORG", "MISC"],
}
NER_CHUNK_SIZE = 10_000           # entries per multiprocessing chunk
NER_N_PROCESSES = 8              # parallel workers
NER_BATCH_SIZE = 512             # spaCy nlp.pipe batch size

# --- Embedding model ---
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBED_DEVICE = "cuda:0"
EMBED_BATCH_SIZE = 128

# --- Stage 1 monolingual filtering ---
KNN_K = 5                        # neighbors for Local Dispersion (Eq. 1)
N_CLUSTERS_PER_LANG = 100        # KMeans clusters per language
TOP_FRACTION = 0.2               # keep top-20% by entropy within each cluster

# --- Stage 2 cross-lingual CP selection ---
CROSS_K = 1000                   # global clusters
TAU = 5                          # min cluster size (Statistical Stability)
THETA = 0.8                      # linguistic dominance threshold (Eq. 4)
