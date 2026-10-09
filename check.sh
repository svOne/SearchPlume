#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON="${SEARCHPLUME_CHECK_PYTHON:-.venv/bin/python}"
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" scripts/metric_controls.py > /dev/null
printf '%s\n' 'Release evidence and official synthetic metric controls passed.'
