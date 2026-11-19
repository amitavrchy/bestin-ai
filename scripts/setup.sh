#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
./scripts/install_rhubarb.sh
.venv/bin/python -c 'from app.providers.registry import Embeddings; Embeddings().embed(["Warm up the model."])'
