#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
cd server
npm install
npm run start:dev
