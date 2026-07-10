#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/docreader-cache/models /tmp/easyocr /tmp/easyocr-reg
chown -R appuser:appuser /tmp/docreader-cache /tmp/easyocr /tmp/easyocr-reg

exec gosu appuser "$@"
