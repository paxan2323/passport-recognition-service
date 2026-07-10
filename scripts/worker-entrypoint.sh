#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/docreader-cache/models /tmp/easyocr /tmp/easyocr-reg
if [ -w /tmp/docreader-cache/models ]; then
  chown -R appuser:appuser /tmp/docreader-cache
fi
chown -R appuser:appuser /tmp/easyocr /tmp/easyocr-reg

exec gosu appuser "$@"
