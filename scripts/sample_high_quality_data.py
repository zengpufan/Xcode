"""Backwards-compatible entry point.

The density + semantic-entropy quality selection now lives inside
``step1_single_lang_clustering.py`` (merged there so document embeddings are
computed exactly once per language instead of twice). This thin wrapper
preserves the old invocation: ``python sample_high_quality_data.py`` simply
runs the merged Stage 1 pipeline.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_single_lang_clustering import run

if __name__ == "__main__":
    run()
