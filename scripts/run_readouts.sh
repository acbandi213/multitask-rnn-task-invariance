#!/usr/bin/env bash
# Level 2: recompute every stored result from the trained networks in models/.
#   scripts/run_readouts.sh            all seven sets of networks, then every figure panel
#   scripts/run_readouts.sh main       only the two main sets (Figure 5, most of Extended Data Figures 7-8)
#   scripts/run_readouts.sh dial       only the five pursuit-dial sets (Extended Data Figure 7 g, right)
#   scripts/run_readouts.sh panels     only the figure panels (needs the read-outs)
# A GPU (CUDA or Apple MPS) is picked automatically; everything also runs on the CPU, more slowly.
set -euo pipefail
cd "$(dirname "$0")/.."
what="${1:-all}"

if [[ "$what" == all || "$what" == main ]]; then
  for run in main_relu main_softplus; do
    python -m mtrnn.evaluate "$run"                     # performance per task, provenance check     -> results/performance/
    python -m mtrnn.readout all "$run"                  # unit features, all 351 task pairs          -> results/readout/
    python -m mtrnn.heldout "$run" --min_perf 0.9       # held-out TI / TD labels and decoding       -> results/readout/
  done
  python -m mtrnn.measured main_relu main_softplus      # the two analog pairs read like recordings  -> results/measured/
fi

if [[ "$what" == all || "$what" == dial ]]; then
  for g in 000 025 050 075 100; do
    run=pursuit_dial_g$g
    python -m mtrnn.evaluate "$run"
    python -m mtrnn.readout features "$run"
    python -m mtrnn.readout measured "$run"
  done
  python -m mtrnn.measured pursuit_dial_g000 pursuit_dial_g025 pursuit_dial_g050 pursuit_dial_g075 pursuit_dial_g100
fi

if [[ "$what" == all || "$what" == panels ]]; then
  python -m mtrnn.panels all                            # one file per figure panel                  -> results/panels/
fi
