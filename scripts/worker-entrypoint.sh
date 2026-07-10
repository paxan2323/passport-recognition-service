#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/docreader-cache/models /tmp/easyocr /tmp/easyocr-reg
if [ -w /tmp/docreader-cache/models ]; then
  chown -R appuser:appuser /tmp/docreader-cache /tmp/easyocr /tmp/easyocr-reg
fi

exec gosu appuser "$@"
