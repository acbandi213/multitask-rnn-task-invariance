#!/usr/bin/env bash
# Maintainer: pack results/ into the two release archives that scripts/download_results.py fetches.
#   dist/results.tar.gz     everything the notebooks read + the small read-out tables (~130 MB)
#   dist/features.tar.gz    simulated unit features of the two main sets (optional; exact recomputation of the CPU panels)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist
tar -czf dist/results.tar.gz --exclude='features.npz' --exclude='results/panels/task_decoding_rates' \
    --exclude='trajectory_features_*.npz' --exclude='silencing_parts' --exclude='heldout_labels_*.pkl' --exclude='.DS_Store' results
tar -czf dist/features.tar.gz results/readout/main_relu/features.npz results/readout/main_softplus/features.npz
shasum -a 256 dist/results.tar.gz dist/features.tar.gz | sed 's# dist/# #' > scripts/release.sha256
ls -lh dist; cat scripts/release.sha256
