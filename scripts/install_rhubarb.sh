#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT
curl -fL --retry 2 -o "$TEMP_DIR/rhubarb.zip" 'https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.14.0/Rhubarb-Lip-Sync-1.14.0-macOS.zip'
unzip -q "$TEMP_DIR/rhubarb.zip" -d "$TEMP_DIR/unpacked"
mkdir -p "$ROOT/tools/bin"
cp -R "$TEMP_DIR/unpacked/Rhubarb-Lip-Sync-1.14.0-macOS/". "$ROOT/tools/bin/"
chmod +x "$ROOT/tools/bin/rhubarb"
"$ROOT/tools/bin/rhubarb" --version
