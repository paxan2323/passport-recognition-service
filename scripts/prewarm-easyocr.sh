#!/usr/bin/env bash
set -euo pipefail

DIR="${1:-/var/lib/docreader/easyocr-en}"
mkdir -p "$DIR"
chown -R 1000:1000 "$DIR"

echo "==> Pre-download EasyOCR EN models into ${DIR}"
docker run --rm \
  -v "${DIR}:/easyocr" \
  -e EASYOCR_MODULE_PATH=/easyocr \
  passport-recognition-service-worker:latest \
  python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=True, model_storage_directory='/easyocr/model')"

echo "==> EasyOCR cache ready"
