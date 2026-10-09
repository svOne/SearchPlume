#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
mkdir -p artifacts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
export HF_HOME="$PWD/cache/huggingface" HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1 DO_NOT_TRACK=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
PYTHON="${SEARCHPLUME_PYTHON:-python3.11}"
"$PYTHON" -c 'import platform,shutil,sys; assert sys.version_info[:3]==(3,11,7), "Recorded run requires Python 3.11.7"; assert platform.system()=="Darwin" and platform.machine()=="arm64", "Only macOS arm64 was tested"; assert int(platform.mac_ver()[0].split(".")[0])>=14; assert shutil.disk_usage(".").free>3*1024**3'
if [ ! -x .venv/bin/python ]; then
    "$PYTHON" -m venv .venv
fi
.venv/bin/python scripts/freeze.py
.venv/bin/python -m pip --isolated install --only-binary=:all: --no-cache-dir --require-hashes --index-url https://pypi.org/simple -r requirements.lock > artifacts/setup.log 2>&1
.venv/bin/python -m pip check
.venv/bin/python scripts/verify_env.py
.venv/bin/python scripts/host.py
.venv/bin/python scripts/prepare.py
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/benchmark.py
.venv/bin/python scripts/compare_reference.py
