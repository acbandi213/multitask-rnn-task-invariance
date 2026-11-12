#!/usr/bin/env bash
# Level 3: retrain the paper's seven sets of networks (10 networks each, trained together; ~3.5 h per set on an Apple M-series GPU).
#
#   scripts/train_all.sh /path/to/fresh_root
#
# The retrained networks and everything computed from them go to <fresh_root>/models and <fresh_root>/results, so the networks that
# ship with the repository are never overwritten. To read the retrained networks afterwards:
#
#   export MTRNN_ROOT=/path/to/fresh_root
#   scripts/run_readouts.sh            # recompute every result
#   (run the notebooks in analysis/ from the same shell)
#
# Seeds fix the initial weights and the trial streams, but GPU arithmetic differs between devices: retrained networks reproduce the
# paper's results statistically, not number for number.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
root="${1:?usage: scripts/train_all.sh /path/to/fresh_root}"
mkdir -p "$root/models" "$root/results"
cp -R "$here/reference" "$root/"
export MTRNN_ROOT="$root"
cd "$here"
for cfg in main_relu main_softplus pursuit_dial_g000 pursuit_dial_g025 pursuit_dial_g050 pursuit_dial_g075 pursuit_dial_g100; do
  python -m mtrnn.train --config "configs/$cfg.json" 2>&1 | tee "$root/models/$cfg.log"
done
