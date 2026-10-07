"""Stage 1 of C-Mining: monolingual high-quality data filtering.

For each language this pipeline:
1. loads the NER-selected entries (from step0),
2. encodes ``<title> T <text> P1`` with a frozen multilingual encoder,
3. KMeans-clusters them into broad thematic domains,
4. within each cluster applies Local Dispersion (Eq. 1) + Semantic Entropy
   (Eq. 2-3) to retain representative, semantically deep candidates, and
5. keeps the top-``TOP_FRACTION`` by entropy as ``C_high`` for that language.

The density + entropy selection that previously lived in
``sample_high_quality_data.py`` is merged here so embeddings are computed
exactly once per language.

Output (per language): ``{SINGLE_LANG_OUTPUT_DIR}/{lang}_high_quality.jsonl``
plus ``{lang}_clusters.json`` for inspection.
"""
import json
import os
import sys
from typing import List

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    entropy_of_similarity,
    first_paragraph,
    format_input,
    get_embedder,
    get_kmeans,
    local_dispersion,
    read_jsonl,
    write_jsonl,
)
from config import (
    DATASET_CACHE_DIR,
    DATASET_DATE,
    DATASET_NAME,
    KNN_K,
    LANG_LIST,
    N_CLUSTERS_PER_LANG,
    NER_OUTPUT_DIR,
    SINGLE_LANG_OUTPUT_DIR,
    TOP_FRACTION,
)


def _load_selected(lang: str):
    """Read NER-selected records and fetch their title + first paragraph."""
    sel_path = os.path.join(NER_OUTPUT_DIR, lang, f"{lang}_selected_data.jsonl")
    records = read_jsonl(sel_path)
    if not records:
        raise RuntimeError(f"No NER-selected entries for {lang} at {sel_path}")
    indices = [r["index"] for r in records]

    data = load_dataset(
        DATASET_NAME, f"{DATASET_DATE}.{lang}", cache_dir=DATASET_CACHE_DIR
    )["train"]
    sub = data.select(indices)
    titles = [r["title"] for r in records]
    paragraphs = [first_paragraph(t) if t else "" for t in sub["text"]]
    return titles, paragraphs, indices


def _semantic_units(text: str) -> list[str]:
    """Paragraphs for entropy. The leading paragraph alone yields H=0, so when
    the article has no second newline-delimited paragraph we fall back to
    sentence splitting to keep the metric meaningful."""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paras) < 2 and len(text) > 40:
        paras = [s.strip() for s in text.replace("。", ".\n").split("\n") if s.strip()]
    return paras if paras else [text]


def _process_lang(lang: str, embedder, out_dir: str) -> None:
    titles, paragraphs, indices = _load_selected(lang)
    inputs = [format_input(t, p) for t, p in zip(titles, paragraphs)]
    embeddings = embedder.encode(
        inputs, batch_size=128, show_progress_bar=True, convert_to_numpy=True
    )

    n_clusters = min(N_CLUSTERS_PER_LANG, len(embeddings))
    labels = get_kmeans(n_clusters).fit_predict(embeddings)

    by_cluster: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        by_cluster.setdefault(int(lab), []).append(i)

    summary = [
        {"cluster_id": c, "size": len(members),
         "titles": [titles[i] for i in members],
         "indices": [indices[i] for i in members]}
        for c, members in sorted(by_cluster.items())
    ]
    with open(os.path.join(out_dir, f"{lang}_clusters.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False)

    high_quality = []
    for c, members in tqdm(by_cluster.items(), desc=f"[{lang}] density+entropy"):
        if len(members) <= KNN_K:
            continue
        cluster_emb = embeddings[members]

        # (1) Local Dispersion δ_m (Eq. 1): keep δ_m ≤ cluster median.
        disp = local_dispersion(cluster_emb, k=KNN_K)
        median = np.median(disp)
        dense_mask = disp <= median
        if not dense_mask.any():
            dense_mask = np.ones(len(disp), dtype=bool)
        # (local_pos, global_idx, disp_value) for kept entries.
        kept = [(j, members[j], float(disp[j]))
                for j in np.where(dense_mask)[0]]

        # (2) Semantic Entropy H(D): batch-encode all kept docs' units at once.
        doc_units = [_semantic_units(paragraphs[gi]) for _, gi, _ in kept]
        flat = [u for units in doc_units for u in units]
        para_emb = (embedder.encode(flat, batch_size=256, show_progress_bar=False,
                                    convert_to_numpy=True)
                    if flat else np.zeros((0, 1), dtype=np.float32))

        offset = 0
        candidates = []
        for (local_pos, gi, d), units in zip(kept, doc_units):
            m = len(units)
            h = entropy_of_similarity(para_emb[offset:offset + m]) if m >= 2 else 0.0
            offset += m
            candidates.append({"local_pos": local_pos, "gi": gi, "density": d, "entropy": h})

        # (3) Keep top-TOP_FRACTION by entropy (paper: "prioritize higher H(D)").
        candidates.sort(key=lambda x: x["entropy"], reverse=True)
        keep_n = max(1, int(len(candidates) * TOP_FRACTION))
        for cand in candidates[:keep_n]:
            high_quality.append({
                "lang": lang,
                "title": titles[cand["gi"]],
                "text": paragraphs[cand["gi"]],
                "index": indices[cand["gi"]],
                "cluster_id": c,
                "density": cand["density"],
                "entropy": cand["entropy"],
            })

    write_jsonl(os.path.join(out_dir, f"{lang}_high_quality.jsonl"), high_quality)
    print(f"[{lang}] C_high = {len(high_quality)} candidates")


def run(lang_list: List[str] = LANG_LIST) -> None:
    if not SINGLE_LANG_OUTPUT_DIR:
        raise SystemExit("config.SINGLE_LANG_OUTPUT_DIR is empty; fill it in config.py")
    os.makedirs(SINGLE_LANG_OUTPUT_DIR, exist_ok=True)
    embedder = get_embedder()  # loaded once, reused across languages
    for lang in lang_list:
        print(f"\n=== Stage 1: {lang} ===")
        _process_lang(lang, embedder, SINGLE_LANG_OUTPUT_DIR)


if __name__ == "__main__":
    run()
