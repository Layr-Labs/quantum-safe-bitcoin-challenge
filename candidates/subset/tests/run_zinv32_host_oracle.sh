#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
python3 candidates/subset/tests/zinv32_host_oracle.py
