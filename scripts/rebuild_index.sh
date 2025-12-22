#!/usr/bin/env bash
set -e
python -m src.preprocessing.clean_events
python -m src.indexing.build_faiss_index
