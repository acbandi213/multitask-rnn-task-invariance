# Trained networks

Seven sets of 10 networks (seeds 0-9), each set trained together by `python -m mtrnn.train --config configs/<name>.json`.
All are 256-unit leaky rate RNNs (Yang et al. 2019) trained on the 27-task battery for 20,000 steps of 64 trials.

| set | used in | what differs |
|---|---|---|
| `main_relu` | Figure 5, Extended Data Figures 7 and 8 | ReLU, all answers on one shared output ring |
| `main_softplus` | Extended Data Figure 7 a, d, e | as `main_relu` with Softplus |
| `pursuit_dial_g000` ... `g100` | Extended Data Figure 7 g (right) | a fraction g (0, 0.25, 0.5, 0.75, 1) of the pursuit answer on the shared ring, the rest on a second ring |

```
<set>/config.json      every setting of the run (the file configs/<set>.json reproduces)
<set>/history.json     performance of every network on every task, every 500 steps
<set>/timing.json      training time and device
<set>/train_log.txt    the training log
<set>/seed<k>/model.pt final weights of network k, with its config hash and a noise-free performance fingerprint
<set>/seed<k>/ckpt_<step>.pt   main_relu only: weights after 1,000 / 2,000 / 5,000 / 10,000 / 20,000 steps (Extended Data Figure 7 h)
```

Load a set:

```python
from mtrnn.train import load_nets
net, battery, cfg, checkpoints = load_nets('models/main_relu')        # net.W_rec has shape (10, 256, 256)
```

`python -m pytest tests/test_shipped_networks.py` checks every file against its config hash and fingerprint.

Names used while the work was in progress, for anyone matching these to older notes: `main_relu` = `q1_relu_shared`,
`main_softplus` = `q1_softplus_shared`, `pursuit_dial_g*` = `pd_relu_partial_g*`.
