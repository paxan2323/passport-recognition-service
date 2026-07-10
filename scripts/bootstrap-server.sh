#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/passport-recognition-service"
REPO_URL="${REPO_URL:-https://github.com/paxan2323/passport-recognition-service.git}"

if ! command -v docker >/dev/null 2>&1; then
  echo "==> Install Docker"
  curl -fsSL https://get.docker.com | sh
  systemctl enable docker
  systemctl start docker
fi

mkdir -p "$APP_DIR"
if [ ! -d "$APP_DIR/.git" ]; then
  git clone "$REPO_URL" "$APP_DIR"
fi

if [ ! -f "$APP_DIR/.env" ]; then
  echo "Create $APP_DIR/.env with PASSPORT_API_KEYS before first deploy"
  cp "$APP_DIR/.env.example" "$APP_DIR/.env"
fi

chmod +x "$APP_DIR/scripts/deploy.sh"
echo "Bootstrap complete. Set secrets in $APP_DIR/.env then run scripts/deploy.sh"
