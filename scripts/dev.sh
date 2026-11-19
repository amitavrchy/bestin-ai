#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
exec .venv/bin/uvicorn app.main:app --port 8001 --reload
