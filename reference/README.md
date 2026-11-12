# Reference values from the recordings

`three_point.csv` holds the only numbers from the neural recordings that this repository needs: across-task choice decoding from
size-matched populations of task-dependent (TD) units, random choice units and task-invariant (TI) units (the paper's Figure 4 h, n;
mean over sessions), for mouse PPC and monkey FEF. Extended Data Figure 7 g compares the networks with these values.

`window` is the mouse analysis window: `pre_2_1` (-2 to -1 s before turn onset) is the paper's; `fig3f` is an earlier window, kept
because the noise sweep was first fitted against it. The monkey values do not depend on it.

Two more constants from the recordings sit in `src/mtrnn/readout.py` (`DATA`): the mean within-task single-neuron accuracy that the
measurement noise is calibrated to (PPC 0.600, FEF 0.565) and the trial counts per condition (50 and 150).
