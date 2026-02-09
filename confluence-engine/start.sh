#!/bin/bash
# Confluence Engine — One-command startup
# Usage: ./start.sh
#
# This pulls the latest code from GitHub, builds/starts all containers
# (PostgreSQL, Redis, API), runs migrations, and seeds the watchlist.
# After this, the dashboard is live at http://localhost:8000

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Confluence Engine Startup ==="
echo ""

# Step 1: Pull latest code
echo "[1/4] Pulling latest code from GitHub..."
cd ..
git pull origin claude/unzip-downloads-folder-KK3gw --rebase 2>/dev/null || \
  git pull origin main --rebase 2>/dev/null || \
  echo "  (skipped — no remote changes or offline)"
cd "$SCRIPT_DIR"
echo ""

# Step 2: Build and start all containers
echo "[2/4] Starting Docker containers (db + redis + api)..."
docker compose up -d --build
echo ""

# Step 3: Wait for API to be ready
echo "[3/4] Waiting for API to come online..."
for i in $(seq 1 30); do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "  API is ready!"
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "  Warning: API took too long to start. Check: docker compose logs api"
    exit 1
  fi
  sleep 2
done
echo ""

# Step 4: Seed watchlist (safe to run multiple times)
echo "[4/4] Seeding watchlist..."
docker compose exec api python3 -m scripts.seed_watchlist
echo ""

echo "=== Confluence Engine is LIVE ==="
echo "Dashboard: http://localhost:8000"
echo "API docs:  http://localhost:8000/docs"
echo ""
echo "Scanner runs every 15 minutes automatically."
echo "To stop: docker compose down"
echo "To view logs: docker compose logs -f api"
