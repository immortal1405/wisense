#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
cd web
npm install
npm run dev
