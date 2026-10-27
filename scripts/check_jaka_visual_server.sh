#!/usr/bin/env bash
# Read-only host preflight for the high-driver JAKA RGB-D collection server.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

failures=0
check_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "FAIL missing command: $1"
    failures=$((failures + 1))
  else
    echo "OK command: $1 -> $(command -v "$1")"
  fi
}

echo "JAKA_VISUAL_SERVER_PREFLIGHT"
echo "repo=$REPO_ROOT"
echo "host=$(hostname)"
echo "arch=$(uname -m)"
echo "kernel=$(uname -r)"

check_command git
check_command docker
check_command nvidia-smi
check_command sha256sum

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
fi

if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    echo "OK docker daemon reachable"
  else
    echo "FAIL docker daemon is not reachable by the current user"
    failures=$((failures + 1))
  fi
fi

available_kb="$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print $4}')"
available_gib=$((available_kb / 1024 / 1024))
echo "workspace_free_gib=$available_gib"
if [[ "$available_gib" -lt 250 ]]; then
  echo "FAIL at least 250 GiB free is required for two full RGB-D shards plus caches"
  failures=$((failures + 1))
else
  echo "OK workspace free space"
fi

if [[ "$failures" -ne 0 ]]; then
  echo "JAKA_VISUAL_SERVER_PREFLIGHT_FAILED count=$failures"
  exit 1
fi
echo "JAKA_VISUAL_SERVER_PREFLIGHT_OK"
