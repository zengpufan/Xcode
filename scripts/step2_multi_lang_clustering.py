"""Stage 2 of C-Mining: multilingual Culture Point selection.

Projects every language's ``C_high`` candidates (from step1) into the shared
multilingual embedding space, re-clusters them into ``CROSS_K`` global groups,
and keeps clusters that are (1) statistically stable (size ≥ τ) and
(2) linguistically dominant (γ(G_j, l*) > θ) — i.e. the geometrically
misaligned "islands" that carry culture-specific semantics (Eq. 4).

Output: ``{MULTI_LANG_OUTPUT_DIR}/culture_points.json``.
"""
import json
import os
import sys
from typing import List

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import format_input, get_embedder, get_kmeans, read_jsonl
from config import (
    CROSS_K,
    LANG_LIST,
    MULTI_LANG_OUTPUT_DIR,
    SINGLE_LANG_OUTPUT_DIR,
    TAU,
    THETA,
)


def _load_high_quality(lang_list: List[str]):
    """Concatenate C_high candidates from all languages."""
    records = []
    for lang in lang_list:
        path = os.path.join(SINGLE_LANG_OUTPUT_DIR, f"{lang}_high_quality.jsonl")
        if not os.path.exists(path):
            print(f"[warn] missing {path}; skipping {lang}")
            continue
        records.extend(read_jsonl(path))
    return records


def _process(records):
    if not records:
        raise RuntimeError("No C_high candidates loaded; run step1 first.")

    inputs = [format_input(r["title"], r["text"]) for r in records]
    embedder = get_embedder()
    embeddings = embedder.encode(
        inputs, batch_size=128, show_progress_bar=True, convert_to_numpy=True
    )

    # No StandardScaler: the paper relies on the raw embedding geometry, and
    # scaling would distort the cross-lingual distances we mine.
    k = min(CROSS_K, len(embeddings))
    labels = get_kmeans(k).fit_predict(embeddings)

    culture_points = []
    for cluster_id in tqdm(range(k), desc="CP selection"):
        members = np.where(labels == cluster_id)[0]
        if len(members) < TAU:
            continue

        langs = [records[i]["lang"] for i in members]
        dist: dict[str, int] = {}
        for lg in langs:
            dist[lg] = dist.get(lg, 0) + 1
        dominant, count = max(dist.items(), key=lambda kv: kv[1])
        gamma = count / len(members)
        if gamma <= THETA:
            continue

        culture_points.append({
            "cluster_id": int(cluster_id),
            "size": int(len(members)),
            "dominant_lang": dominant,
            "dominance": round(gamma, 3),
            "lang_distribution": dist,
            "samples": [
                {"title": records[i]["title"], "text": records[i]["text"],
                 "lang": records[i]["lang"], "index": records[i]["index"]}
                for i in members
            ],
        })

    # Sort by dominance then size for stable, readable output.
    culture_points.sort(key=lambda c: (c["dominance"], c["size"]), reverse=True)
    return culture_points


def run(lang_list: List[str] = LANG_LIST) -> None:
    if not MULTI_LANG_OUTPUT_DIR:
        raise SystemExit("config.MULTI_LANG_OUTPUT_DIR is empty; fill it in config.py")
    os.makedirs(MULTI_LANG_OUTPUT_DIR, exist_ok=True)

    records = _load_high_quality(lang_list)
    print(f"Loaded {len(records)} C_high candidates across {lang_list}")
    culture_points = _process(records)

    out_path = os.path.join(MULTI_LANG_OUTPUT_DIR, "culture_points.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(culture_points, f, ensure_ascii=False, indent=2)

    # Summary by language.
    by_lang: dict[str, int] = {}
    for cp in culture_points:
        by_lang[cp["dominant_lang"]] = by_lang.get(cp["dominant_lang"], 0) + 1
    print(f"\nCulture Points: {len(culture_points)} clusters → {out_path}")
    for lg, n in sorted(by_lang.items()):
        print(f"  {lg}: {n}")


if __name__ == "__main__":
    run()
