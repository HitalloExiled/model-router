#!/usr/bin/env bash
# model-router check: syntax + offline unit tests (no GPU/docker/network).
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python3 -m py_compile "$root/ninfer-router.py"
python3 "$root/tests/test_router.py"

printf '%s\n' 'model-router checks passed.'
