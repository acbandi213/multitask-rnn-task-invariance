# Multitask RNNs: task-invariant and task-dependent units

Code, trained networks and figure notebooks for the recurrent-network part of the paper (Figure 5, Extended Data Figures 7 and 8).

Leaky rate RNNs in the style of Yang et al. (2019, *Nat. Neurosci.*) are trained on 27 tasks at once: Yang's 20 tasks plus 7 tasks
modeled on the experiments (for example a memory-guided saccade and a pursuit task, as recorded in monkey FEF, and two tasks that
give the same answer from different input modalities, as in mouse PPC). Every unit is then read the way the recorded neurons were:
for each pair of tasks, does the unit carry the choice in the same way in both (task-invariant, TI) or not (task-dependent, TD), and
what does that mix do to decoding the choice across tasks?

## Layout

```
src/mtrnn/          the package
  tasks.py            the 27-task battery (trial parameters in NumPy, rendering in PyTorch)
  model.py            the network; K networks with different seeds train together as one batched module
  train.py            training, checkpoints, provenance check
  evaluate.py         performance per task and network
  readout.py          unit features, the cross-task generalization index for all 351 task pairs
  heldout.py          TI / TD labels from one half of the trials, decoding on the other half
  measured.py         the two pairs that mirror the experiments, read with the recordings' trial counts and noise
  silencing.py        silencing units in the running network
  panels/             one function per panel of Extended Data Figures 7 (robustness.py) and 8 (roles.py)
  lib/                the single-unit index and decoders, copied unchanged from the analysis of the recordings
configs/            one JSON file per set of networks
models/             the seven trained sets of 10 networks (see models/README.md)
results/            stored results; downloaded or recomputed, not in git
reference/          the six values from the recordings that the networks are compared with
analysis/           figure_5.ipynb, figure_s7.ipynb, figure_s8.ipynb (Extended Data Figures 7, 8), with their output
scripts/            download, recompute and retrain
tests/
```

## Install

Python 3.11. From the repository folder:

```bash
conda env create -f environment.yml
```
```bash
conda activate mtrnn
```

or, in an existing environment, `pip install -r requirements.txt && pip install -e .`

`requirements.txt` pins the versions the results were made with. The stored panel results are pandas pickles, so pandas 2.0.x is
needed to read them; the other pins matter less.

## Three ways to reproduce

### 1. The figures, from stored results (about a minute per notebook, no GPU)

```bash
python scripts/download_results.py
```
```bash
cd analysis && jupyter nbconvert --to notebook --execute --inplace figure_5.ipynb figure_s7.ipynb figure_s8.ipynb
```

or open the notebooks in Jupyter from the `analysis/` folder. Panels are written to `analysis/figures/<figure>/` and the statistics
of every panel to `analysis/tables/` (both are committed, so `git diff` shows whether anything changed). While the repository is
private the download uses the GitHub command-line tool (`gh auth login` once).

### 2. The results, from the trained networks

```bash
scripts/run_readouts.sh
```

This runs, for each set of networks, `evaluate` → `readout` → `heldout` → `measured`, and then every figure panel
(`python -m mtrnn.panels all`). `scripts/run_readouts.sh main|dial|panels` runs one part; `python -m mtrnn.panels` lists the panels
and what each needs. A GPU (CUDA or Apple MPS) is used when there is one; everything also runs on the CPU. On an Apple M-series
laptop the panel step takes about 20 min for the CPU panels, 3 min for the simulations and 40 min for silencing.

The read-outs simulate new trials with private noise that is not seeded (the trial parameters are), so recomputed numbers agree
with the stored ones to within trial noise, not digit for digit. To recompute the panels that start from unit features exactly,
download the stored features first:

```bash
python scripts/download_results.py --features
```

### 3. The networks, from scratch

```bash
scripts/train_all.sh /path/to/fresh_root
```

Retrains the seven sets (about 3.5 h per set of 10 networks on an Apple M-series GPU) into `<fresh_root>/models`, never over the
networks that ship here. Then `export MTRNN_ROOT=/path/to/fresh_root`, run `scripts/run_readouts.sh` and the notebooks. Seeds fix
the initial weights and the trial streams, but GPU arithmetic differs between devices, so retrained networks reproduce the results
statistically, not number for number.

## Training new networks

Copy `configs/example_new.json`, give it a new `name` and change what you need; any setting in `DEFAULTS` of `src/mtrnn/train.py`
can be set (units, activation, output layout, regularization, which tasks, seeds, ...).

```bash
python -m mtrnn.train --config configs/my_networks.json
```

The networks go to `models/<name>/`; training refuses to overwrite an existing set. The same read-outs then apply:

```bash
python -m mtrnn.evaluate my_networks
```
```bash
python -m mtrnn.readout all my_networks
```
```bash
python -m mtrnn.heldout my_networks --min_perf 0.9
```

which write `results/performance/my_networks/` and `results/readout/my_networks/` (`units.parquet`: one row per unit, task pair
and network; `pairs.parquet` and `heldout_pairs_p90.parquet`: one row per task pair and network). `analysis/figure_5.ipynb` reads
exactly these three tables: set `RUN = 'my_networks'` in its first cell to draw Figure 5 for the new networks. The panel functions
of the Extended Data figures are written for `main_relu` / `main_softplus` (`RUN`, `RUN_SP` in `src/mtrnn/panels/common.py`).

New tasks are added in `src/mtrnn/tasks.py` (one generator function per task family, registered in `Battery`).

## Tests

```bash
python -m pytest
```

checks the task battery, a short training run, every shipped network against its stored config hash and performance fingerprint,
that each file in `configs/` reproduces the settings of the shipped set, and that the stored results are complete and consistent.
`python -m pytest --slow` also recomputes two panels from the networks and the stored features and compares them with the stored
results.

## Terms

| in the code | meaning |
|---|---|
| `snCCGP` | cross-task generalization index of one unit: a choice threshold fitted in one task, scored in the other (0.5 = chance, below 0.5 = reversed preference) |
| `sn_within` | the same threshold scored within a task (cross-validated) |
| `group` | choice units of a task pair: active units with within-task accuracy ≥ 0.55 in either task |
| TI / TD | top / bottom 20 % of the index, pooled over task pairs and networks |
| `within_cd`, `across_cd` | population choice decoding within a task, and trained on one task and tested on the other |
| pair types | same operation, different operation, spatial vs abstract response, reversed mapping (`readout.relation`) |
| `mouse_like`, `monkey_like` | the task pairs that mirror the experiments: Go Mod1 vs Go Mod2, and Sac vs Pursuit |

## License

MIT (see `LICENSE`).
