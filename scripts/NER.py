"""Stage 0 of C-Mining: NER-based negative filtering on Wikipedia.

Drops entries whose title/text are dominated by functional NER categories
(dates, measurements, etc.) unlikely to carry cultural semantics, and keeps
articles whose title co-occurs with a relevant named entity. Uses spaCy
``nlp.pipe`` for batched processing and multiprocessing across chunks.

Output (per language): ``{NER_OUTPUT_DIR}/{lang}/{lang}_selected_data.jsonl``
and ``{lang}_filtered_data.jsonl``.
"""
import json
import multiprocessing
import os
import string
import sys
from typing import List

import spacy
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATASET_CACHE_DIR,
    DATASET_DATE,
    DATASET_NAME,
    LANG_LIST,
    LANG_SAMPLE_SIZE,
    NER_BATCH_SIZE,
    NER_CHUNK_SIZE,
    NER_N_PROCESSES,
    NER_OUTPUT_DIR,
    NER_VALID_ENTITIES,
    SPACY_MODEL,
)


def _valid_entities(lang: str) -> List[str]:
    return NER_VALID_ENTITIES.get(lang, NER_VALID_ENTITIES["default"])


def _sample_size(lang: str) -> int:
    return LANG_SAMPLE_SIZE.get(lang, LANG_SAMPLE_SIZE["default"])


def _title_is_clean(title: str) -> bool:
    return not any(c.isdigit() or c in string.punctuation for c in title)


def _title_entity_match(doc, title: str, valid: List[str]) -> bool:
    if any(c.isdigit() for c in title):
        return False
    for ent in doc.ents:
        t = ent.text.strip()
        if t and (t in title or title in t) and ent.label_ in valid:
            return True
    return False


def _process_chunk(args) -> None:
    lang, start, end, out_dir = args
    nlp = spacy.load(SPACY_MODEL[lang])

    data = load_dataset(
        DATASET_NAME, f"{DATASET_DATE}.{lang}", cache_dir=DATASET_CACHE_DIR
    )["train"]
    end = min(end, len(data))
    sub = data.select(range(start, end))
    titles = list(sub["title"])
    texts = [t.split("\n")[0].strip() if t else "" for t in sub["text"]]

    # Only run the (expensive) NER on non-empty entries; keep the local offset.
    pending = [
        (i, title, text)
        for i, (title, text) in enumerate(zip(titles, texts))
        if title.strip() and text
    ]
    pipe_texts = [t for _, _, t in pending]

    selected_path = os.path.join(out_dir, f"selected_data_{start}.jsonl")
    filtered_path = os.path.join(out_dir, f"filtered_data_{start}.jsonl")
    n_sel = n_fil = 0
    valid = _valid_entities(lang)
    with open(selected_path, "w", encoding="utf-8") as fs, open(
        filtered_path, "w", encoding="utf-8"
    ) as ff:
        for (local_i, title, text), doc in zip(
            pending, nlp.pipe(pipe_texts, batch_size=NER_BATCH_SIZE)
        ):
            entities = [
                (e.text.strip(), e.label_) for e in doc.ents if e.text.strip()
            ]
            keep = _title_is_clean(title) and _title_entity_match(doc, title, valid)
            rec = {
                "title": title,
                "index": start + local_i,
                "entities": entities,
                "lang_code": lang,
            }
            out = fs if keep and entities else ff
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if keep and entities:
                n_sel += 1
            else:
                n_fil += 1
    print(f"[{lang} {start}-{end}] selected={n_sel} filtered={n_fil}")


def _merge_chunks(lang: str, out_dir: str) -> None:
    files = [f for f in os.listdir(out_dir) if f.endswith(".jsonl")]
    for prefix, name in [("selected_data_", f"{lang}_selected_data.jsonl"),
                          ("filtered_data_", f"{lang}_filtered_data.jsonl")]:
        chunks = sorted(f for f in files if f.startswith(prefix))
        dst = os.path.join(out_dir, name)
        n = 0
        with open(dst, "w", encoding="utf-8") as out:
            for fname in chunks:
                with open(os.path.join(out_dir, fname), encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            out.write(line)
                            n += 1
                os.remove(os.path.join(out_dir, fname))
        print(f"[{lang}] merged {n} records → {dst}")


def run(lang_list: List[str] = LANG_LIST) -> None:
    if not NER_OUTPUT_DIR:
        raise SystemExit("config.NER_OUTPUT_DIR is empty; fill it in config.py")
    for lang in lang_list:
        out_dir = os.path.join(NER_OUTPUT_DIR, lang)
        os.makedirs(out_dir, exist_ok=True)
        total = _sample_size(lang)
        chunks = [
            (lang, i, min(i + NER_CHUNK_SIZE, total), out_dir)
            for i in range(0, total, NER_CHUNK_SIZE)
        ]
        print(f"[{lang}] {total} entries in {len(chunks)} chunks, "
              f"{NER_N_PROCESSES} workers")
        multiprocessing.set_start_method("spawn", force=True)
        with multiprocessing.Pool(NER_N_PROCESSES) as pool:
            for _ in tqdm(
                pool.imap_unordered(_process_chunk, chunks),
                total=len(chunks),
                desc=f"NER {lang}",
            ):
                pass
        _merge_chunks(lang, out_dir)


if __name__ == "__main__":
    run()
