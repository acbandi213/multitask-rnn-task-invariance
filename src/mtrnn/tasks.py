"""The extended Yang et al. (2019) task battery: Yang's 20 tasks plus 7 data-anchor tasks, in PyTorch-friendly form.

Two layers:
    1. Trial PARAMETERS (numpy, per trial): epoch times, stimulus events, the required response. One function per task family,
       `gen(rng, n, test)`; `test=True` gives the analysis setting (fixed canonical timing, stimulus directions on the 12-direction
       grid 15, 45, ..., 345 deg, so no direction or response sits on the hemifield boundary at 90 / 270 deg).
    2. RENDERING (torch, on the training device): parameters -> inputs (B, T, I), targets (B, T, O), loss mask (B, T, O).
       Rendering on the GPU avoids building ~100 MB of numpy inputs per iteration.

Input  (I = 1 + 3 x 32 + n_rules): fixation | ring mod 1 | ring mod 2 | ring motion | one-hot rule units.
    Rings as in Yang: unit i (preferred direction psi_i = 2 pi i / 32) gets gamma * 0.8 exp(-0.5 (8 |psi_i - psi| / pi)^2).
    The motion ring is new: only the pursuit anchor uses it (target velocity direction, the MT-like input of the step-ramp task).
Output (O = 1 + 2 x 32): fixation | response ring A | response ring B. Background tasks always answer on ring A; ring B is the
    second effector of the anchors under layout 'separate' (pursuit, Cat-effector B), never used under layout 'shared'. The
    architecture is identical in both layouts. Targets / mask follow Yang: fixation 0.85 until the response, then 0.05; ring
    0.05 baseline, bump + 0.05 at the response direction during the response; mask 1 before the response cue, 0 for a 100 ms
    grace period, 5 afterwards, x2 for the fixation output; padded steps 0.
Performance (Yang's rule, last time step): respond trials correct iff fixation output < 0.5 and the population-vector direction of the
    response ring is within 36 deg of the target; no-go trials correct iff fixation output > 0.5.

Times are in ms in the definitions and converted to dt = 20 ms steps. Timings Yang's Methods give are used as given; the rest
(fixation epochs, the post-stimulus delay of the Dly DM tasks) follow the reference code (github.com/gyyang/multitask) as closely
as its documented behaviour allows and are marked `# not in Methods`.

Anchor tasks (the pairs the data measured):
    modality  Go Mod1 / Go Mod2      Yang's Go with the cue fixed to ring 1 / ring 2                     -> mouse PPC / M2
    movement  Sac / Pursuit          memory-guided saccade (target flash, delay, go) vs step-ramp
                                     pursuit (at go the target steps 0.5 forward or backward and ramps in
                                     its motion direction; motion ring = direction; respond with the
                                     motion direction)                                                    -> macaque FEF
    task      Cat / Match            same sample-delay-test stimuli on ring 1; Cat: is the test in the
                                     upper half (0-180 deg)?  Match: is the test in the same half as the
                                     sample? Answer yes = 0 deg (right), no = 180 deg (left)                -> human MFC, task
    effector  Cat / Cat-B            the same Cat task answered on ring A vs ring B                       -> human MFC, effector

Optional additions (2026-10-26; off by default, so the Fig. 5 batteries are unchanged):
    mouse_tasks=True   two faithful T-maze tasks sharing ONE rule unit (the task is told only by the cue, as in the mouse blocks):
                       T-maze Vis  grating 0 deg -> left (180 deg), 90 deg -> right (0 deg), on a 4th input ring (orientation,
                                   x 2 on the circle) that no other task uses, so the cue-to-side map is arbitrary
                       T-maze Aud  sound at -90 deg (left, ring angle 180 deg) or +90 deg (right, 0 deg) on ring mod 2; answer toward it
                       Both: two cue values, two answers, Go Mod1 timing (cue on until the end, go at fixation offset).
    layout='partial'   pursuit's target bump is split: a fraction `pursuit_share` g on ring A (the eye ring of every other task)
                       and 1 - g on ring B; performance from the sum of the two rings. g = 0 is 'separate', g = 1 is 'shared'.
                       Cat-B answers on ring A, as in 'shared'. g = 0 / 1 give the two endpoints in the same battery.
"""
from __future__ import annotations

import numpy as np
import torch

DT = 20
N_RING = 32
N_IN_RINGS = 3                      # mod 1, mod 2, motion
N_OUT_RINGS = 2                     # A, B
PREF = np.arange(N_RING) * 2 * np.pi / N_RING
E_MAX = 4                           # stimulus events per trial (Ctx DM: 2 stimuli x 2 modalities)
GRID = np.deg2rad(15 + 30 * np.arange(12))          # analysis directions: never on 90 / 270 deg
DMC_SET = np.deg2rad([18, 54, 90, 126, 162, 198, 234, 270, 306, 342])   # Yang's DMC training directions
GRACE = 5                           # 100 ms grace period after the response cue
RESP_MS = 500                       # response epoch


def st(ms):
    return np.round(np.asarray(ms, float) / DT).astype(int)


# ═══ trial-parameter builder ═════════════════════════════════════════════════════════════════════════════════════════════
def _new(n):
    z = lambda dt=float: np.zeros(n, dt)
    return dict(T=z(int), fix_off=z(int), resp_on=z(int), respond=np.ones(n, bool), resp_dir=z(), resp_ring=z(int), resp_share=z(),
                ev_ring=np.zeros((n, E_MAX), int), ev_dir=np.zeros((n, E_MAX)), ev_s0=np.zeros((n, E_MAX)),
                ev_slope=np.zeros((n, E_MAX)), ev_on=np.zeros((n, E_MAX), int), ev_off=np.zeros((n, E_MAX), int),
                n_ev=z(int), meta={})


def _ev(b, ring, d, s0, on, off, slope=0.0, m=None):
    """Add one stimulus event to the trials in boolean mask m (all if None); every value scalar or per-trial (n,)."""
    n = len(b['T'])
    idx = np.arange(n) if m is None else np.flatnonzero(m)
    take = lambda v: (np.asarray(v)[idx] if np.ndim(v) else np.full(len(idx), v))
    k = b['n_ev'][idx]
    assert (k < E_MAX).all()
    b['ev_ring'][idx, k] = take(ring); b['ev_dir'][idx, k] = take(d) % (2 * np.pi); b['ev_s0'][idx, k] = take(s0)
    b['ev_slope'][idx, k] = take(slope); b['ev_on'][idx, k] = take(on); b['ev_off'][idx, k] = take(off)
    b['n_ev'][idx] += 1


def _u(rng, n, a, b_, test, fixed):
    return st(np.full(n, fixed)) if test else st(rng.uniform(a, b_, n))


def _c(rng, n, opts, test, fixed):
    return st(np.full(n, fixed)) if test else st(rng.choice(opts, n))


def _dirs(rng, n, test):
    """Training: uniform on the circle (Yang). Analysis: the 12-direction grid, balanced (cycled, shuffled)."""
    if not test:
        return rng.uniform(0, 2 * np.pi, n)
    return GRID[rng.permutation(np.resize(np.arange(12), n))]


# ═══ task families ══════════════════════════════════════════════════════════════════════════════════════════════════════
def go_family(kind, anti=False, mod=None, ring_out=0):
    """kind 'fd' (Go: stimulus stays on, respond at fixation offset), 'rt' (RT Go: fixation never goes off, respond at stimulus
    onset), 'dly' (Dly Go: brief stimulus, delay, respond at fixation offset). mod None = random modality 1 / 2 (Yang)."""
    def gen(rng, n, test=False):
        b = _new(n)
        d = _dirs(rng, n, test)
        m = rng.integers(0, 2, n) if mod is None else np.full(n, mod)
        if kind == 'fd':
            on = _u(rng, n, 300, 700, test, 500)                                    # not in Methods
            fix_off = on + _u(rng, n, 500, 1500, test, 1000)
            T = fix_off + st(RESP_MS); resp_on = fix_off; off = T
        elif kind == 'rt':
            on = _u(rng, n, 500, 2500, test, 1000)
            T = on + st(RESP_MS); fix_off = T; resp_on = on; off = T
        else:
            on = _c(rng, n, [300, 500, 700], test, 500)                             # not in Methods
            off = on + _c(rng, n, [200, 400, 600], test, 400)                       # not in Methods
            fix_off = off + _c(rng, n, [200, 400, 800, 1600], test, 800)
            T = fix_off + st(RESP_MS); resp_on = fix_off
        b['T'][:], b['fix_off'][:], b['resp_on'][:] = T, fix_off, resp_on
        _ev(b, m, d, 1.0, on, off)
        b['resp_dir'][:] = (d + (np.pi if anti else 0.0)) % (2 * np.pi); b['resp_ring'][:] = ring_out
        b['meta'].update(stim_dir=d, mod=m)
        return b
    return gen


def dm_family(variant, delayed=False, easy=False):
    """variant 'dm1', 'dm2' (two stimuli in one modality), 'ctx1', 'ctx2' (both modalities, attend one), 'multi' (integrate both).
    One coherence c per trial sets the net evidence for stimulus 1; respond to the stronger stimulus. easy = the reference code's
    `easy_task` option: coherences x 10 (DM family) or x 2 (Dly DM family)."""
    cohs = np.array([0.08, 0.16, 0.32]) if delayed else np.array([0.01, 0.02, 0.04, 0.08])
    if easy:
        cohs = cohs * (2 if delayed else 10)

    def gen(rng, n, test=False):
        b = _new(n)
        d1 = _dirs(rng, n, test)
        d2 = d1 + (np.pi if test else rng.uniform(0.5 * np.pi, 1.5 * np.pi, n))
        coh = lambda: rng.choice(cohs, n) * rng.choice([-1, 1], n)
        if not delayed:
            on = _u(rng, n, 100, 400, test, 300)                                    # not in Methods
            fix_off = on + _c(rng, n, [400, 800, 1600], test, 800)
            T = fix_off + st(RESP_MS)
            w1 = (on, T); w2 = (on, T)                                               # both stimuli on till the end
        else:
            on1 = _u(rng, n, 100, 400, test, 300)                                   # not in Methods
            off1 = on1 + st(300)
            on2 = off1 + _c(rng, n, [200, 400, 800, 1600], test, 800)
            off2 = on2 + st(300)
            fix_off = off2 + _u(rng, n, 100, 300, test, 200)                        # not in Methods
            T = fix_off + st(RESP_MS)
            w1 = (on1, off1); w2 = (on2, off2)
        b['T'][:], b['fix_off'][:], b['resp_on'][:] = T, fix_off, fix_off
        if variant in ('dm1', 'dm2'):
            ring = 0 if variant == 'dm1' else 1
            g = rng.uniform(0.8, 1.2, n); c = coh()
            _ev(b, ring, d1, g + c, *w1); _ev(b, ring, d2, g - c, *w2)
            cm = dict(coh=c)
        elif variant in ('ctx1', 'ctx2'):
            c1, c2 = coh(), coh(); g1, g2 = rng.uniform(0.8, 1.2, n), rng.uniform(0.8, 1.2, n)
            _ev(b, 0, d1, g1 + c1, *w1); _ev(b, 0, d2, g1 - c1, *w2)
            _ev(b, 1, d1, g2 + c2, *w1); _ev(b, 1, d2, g2 - c2, *w2)
            c = c1 if variant == 'ctx1' else c2
            cm = dict(coh=c, coh_mod1=c1, coh_mod2=c2)
        else:
            c = coh(); g = rng.uniform(0.8, 1.2, n)
            ga, gb = g + c, g - c                                                    # mean strength of stimulus 1 / 2 over modalities
            dlt = lambda: rng.uniform(0.1, 0.4, n) * rng.choice([-1, 1], n)
            da, db = dlt(), dlt()
            _ev(b, 0, d1, ga * (1 + da), *w1); _ev(b, 1, d1, ga * (1 - da), *w1)
            _ev(b, 0, d2, gb * (1 + db), *w2); _ev(b, 1, d2, gb * (1 - db), *w2)
            cm = dict(coh=c)
        b['resp_dir'][:] = np.where(c > 0, d1, d2) % (2 * np.pi)
        b['meta'].update(stim1_dir=d1 % (2 * np.pi), stim2_dir=d2 % (2 * np.pi), **cm)
        return b
    return gen


def match_family(category, nogo, min_offset_deg=10.0):
    """DMS / DNMS (category False) and DMC / DNMC (category True). Sample (300 ms), delay, test stays on; fixation input never
    goes off and the response (toward the test) is due at test onset on match (DMS, DMC) or non-match (DNMS, DNMC) trials.
    DMS / DNMS non-match tests lie min_offset_deg ... 360 - min_offset_deg from the sample (Methods: 10; reference code: 20)."""
    def gen(rng, n, test=False):
        b = _new(n)
        m1, m2 = rng.integers(0, 2, n), rng.integers(0, 2, n)
        match = rng.permutation(np.resize([True, False], n))
        if category:
            pool = GRID if test else DMC_SET
            d1 = pool[rng.integers(0, len(pool), n)]
            same = pool[pool < np.pi], pool[pool >= np.pi]
            cat1 = (d1 >= np.pi).astype(int)
            want = np.where(match, cat1, 1 - cat1)
            d2 = np.array([rng.choice(same[w]) for w in want])
        else:
            d1 = _dirs(rng, n, test)
            off = (rng.integers(1, 12, n) * np.deg2rad(30)) if test else rng.uniform(np.deg2rad(min_offset_deg), np.deg2rad(360 - min_offset_deg), n)
            d2 = np.where(match, d1, d1 + off) % (2 * np.pi)
        on1 = _u(rng, n, 100, 400, test, 300)                                       # not in Methods
        off1 = on1 + st(300)
        on2 = off1 + _c(rng, n, [200, 400, 800, 1600], test, 800)
        T = on2 + st(RESP_MS)
        b['T'][:], b['fix_off'][:], b['resp_on'][:] = T, T, on2
        _ev(b, m1, d1, 1.0, on1, off1); _ev(b, m2, d2, 1.0, on2, T)
        b['respond'][:] = ~match if nogo else match
        b['resp_dir'][:] = d2
        b['meta'].update(stim1_dir=d1, stim2_dir=d2, match=match.astype(int), mod1=m1, mod2=m2)
        return b
    return gen


def pursuit(ring_out, p_forward=0.5, step=0.5, ramp_per_step=0.05, share=0.0):
    """Step-ramp pursuit . Fixation only until fixation offset; then the target
    steps `step` forward (along its motion) or backward and ramps at `ramp_per_step` per 20 ms in the motion direction theta.
    Position ring (mod 1): bump at the target's direction from the fixation point, strength = eccentricity (backward steps start at
    theta + 180 deg and cross over to theta). Motion ring: theta, strength 1. Response: theta, from target motion onset.
    share > 0 (layout 'partial'): that fraction of the response bump goes on ring A, the rest on ring_out."""
    def gen(rng, n, test=False):
        b = _new(n)
        th = _dirs(rng, n, test)
        fwd = rng.permutation(np.resize([True, False], n)) if test else rng.random(n) < p_forward
        fix_off = _u(rng, n, 500, 1500, test, 1000)
        T = fix_off + st(RESP_MS)
        b['T'][:], b['fix_off'][:], b['resp_on'][:] = T, fix_off, fix_off
        cross = fix_off + int(round(step / ramp_per_step))
        _ev(b, 0, th, step, fix_off, T, ramp_per_step, m=fwd)
        _ev(b, 0, th + np.pi, step, fix_off, cross, -ramp_per_step, m=~fwd)
        _ev(b, 0, th, 0.0, cross, T, ramp_per_step, m=~fwd)
        _ev(b, 2, th, 1.0, fix_off, T)
        b['resp_dir'][:] = th; b['resp_ring'][:] = ring_out; b['resp_share'][:] = share
        b['meta'].update(stim_dir=th, step_forward=fwd.astype(int))
        return b
    return gen


def two_afc(rule, ring_out=0):
    """Task / effector anchors. Sample and test on ring 1 (Yang's DMC directions in training, the grid in analysis), test stays on,
    a 500 ms hold (300-700 ms in training), then fixation offset = go. rule 'cat': yes iff the test is in the upper half
    (0-180 deg); 'match': yes iff test and sample are in the same half. yes -> 0 deg (right), no -> 180 deg (left)."""
    def gen(rng, n, test=False):
        b = _new(n)
        pool = GRID if test else DMC_SET
        if test:                                                                     # balanced over (sample, test) category cells
            cells = rng.permutation(np.resize(np.arange(4), n)); c1, c2 = cells // 2, cells % 2
            half = [pool[pool < np.pi], pool[pool >= np.pi]]
            d1 = np.array([rng.choice(half[c]) for c in c1]); d2 = np.array([rng.choice(half[c]) for c in c2])
        else:
            d1, d2 = pool[rng.integers(0, len(pool), n)], pool[rng.integers(0, len(pool), n)]
        on1 = _u(rng, n, 100, 400, test, 300)
        off1 = on1 + st(300)
        on2 = off1 + _c(rng, n, [200, 400, 800, 1600], test, 800)
        fix_off = on2 + _u(rng, n, 300, 700, test, 500)
        T = fix_off + st(RESP_MS)
        b['T'][:], b['fix_off'][:], b['resp_on'][:] = T, fix_off, fix_off
        _ev(b, 0, d1, 1.0, on1, off1); _ev(b, 0, d2, 1.0, on2, T)
        up1, up2 = d1 < np.pi, d2 < np.pi
        yes = up2 if rule == 'cat' else (up1 == up2)
        b['resp_dir'][:] = np.where(yes, 0.0, np.pi); b['resp_ring'][:] = ring_out
        b['meta'].update(stim1_dir=d1, stim2_dir=d2, yes=yes.astype(int))
        return b
    return gen


def tmaze(kind):
    """The faithful mouse T-maze tasks (module docstring). kind 'vis': orientation ring (ring 3), grating 0 deg (ring angle 0) ->
    left, 90 deg (ring angle 180 deg) -> right. kind 'aud': sound location on ring mod 2, answer toward it. Go Mod1 timing."""
    def gen(rng, n, test=False):
        b = _new(n)
        right = rng.permutation(np.resize([0, 1], n)).astype(bool) if test else rng.random(n) < 0.5
        on = _u(rng, n, 300, 700, test, 500)
        fix_off = on + _u(rng, n, 500, 1500, test, 1000)
        T = fix_off + st(RESP_MS)
        b['T'][:], b['fix_off'][:], b['resp_on'][:] = T, fix_off, fix_off
        if kind == 'vis':
            cue = np.where(right, np.pi, 0.0)                    # 90 deg grating -> right, 0 deg -> left
            _ev(b, 3, cue, 1.0, on, T)
        else:
            cue = np.where(right, 0.0, np.pi)                    # sound on the right (+90 deg) -> right
            _ev(b, 1, cue, 1.0, on, T)
        b['resp_dir'][:] = np.where(right, 0.0, np.pi); b['resp_ring'][:] = 0
        b['meta'].update(stim_dir=cue, right=right.astype(int))
        return b
    return gen


# ═══ the battery ════════════════════════════════════════════════════════════════════════════════════════════════════════
YANG20 = ['Go', 'RT Go', 'Dly Go', 'Anti', 'RT Anti', 'Dly Anti', 'DM 1', 'DM 2', 'Ctx DM 1', 'Ctx DM 2', 'MultSen DM',
          'Dly DM 1', 'Dly DM 2', 'Ctx Dly DM 1', 'Ctx Dly DM 2', 'MultSen Dly DM', 'DMS', 'DNMS', 'DMC', 'DNMC']
ANCHORS = ['Go Mod1', 'Go Mod2', 'Sac', 'Pursuit', 'Cat', 'Match', 'Cat-B']
MOUSE_TASKS = ['T-maze Vis', 'T-maze Aud']
MOUSE_PAIR = ('T-maze Vis', 'T-maze Aud')        # the faithful mouse pair (not in ANCHOR_PAIRS, so older read-outs are unchanged)
ANCHOR_PAIRS = {'modality': ('Go Mod1', 'Go Mod2'), 'movement': ('Sac', 'Pursuit'),
                'task': ('Cat', 'Match'), 'effector': ('Cat', 'Cat-B')}
ANCHOR_DATA = {'modality': 'mouse PPC / M2', 'movement': 'macaque FEF', 'task': 'human MFC (task)', 'effector': 'human MFC (effector)'}
FAMILY = {**{t: 'Go' for t in YANG20[:3]}, **{t: 'Anti' for t in YANG20[3:6]}, **{t: 'DM' for t in YANG20[6:11]},
          **{t: 'Dly DM' for t in YANG20[11:16]}, **{t: 'Matching' for t in YANG20[16:]}, **{t: 'Anchor' for t in ANCHORS + MOUSE_TASKS}}


class Battery:
    """The task list for one ring layout. layout 'separate': Pursuit and Cat-B answer on ring B; 'shared': everything on ring A;
    'partial': Pursuit split between rings A (fraction pursuit_share) and B, Cat-B on ring A. mouse_tasks adds the two T-maze tasks.
    render_kw (empty for the Fig. 5 batteries) must be passed to render()."""

    def __init__(self, layout='separate', tasks=None, ctx_weight=5.0, easy_dm=False, dms_min_offset_deg=10.0, pursuit_share=None,
                 mouse_tasks=False):
        assert layout in ('separate', 'shared', 'partial')
        assert (layout == 'partial') == (pursuit_share is not None), 'pursuit_share goes with layout partial only'
        self.layout, self.easy_dm = layout, bool(easy_dm)
        e, mo = self.easy_dm, float(dms_min_offset_deg)
        rb = 1 if layout == 'separate' else 0
        gens = {
            'Go': go_family('fd'), 'RT Go': go_family('rt'), 'Dly Go': go_family('dly'),
            'Anti': go_family('fd', True), 'RT Anti': go_family('rt', True), 'Dly Anti': go_family('dly', True),
            'DM 1': dm_family('dm1', easy=e), 'DM 2': dm_family('dm2', easy=e), 'Ctx DM 1': dm_family('ctx1', easy=e), 'Ctx DM 2': dm_family('ctx2', easy=e),
            'MultSen DM': dm_family('multi', easy=e),
            'Dly DM 1': dm_family('dm1', True, easy=e), 'Dly DM 2': dm_family('dm2', True, easy=e), 'Ctx Dly DM 1': dm_family('ctx1', True, easy=e),
            'Ctx Dly DM 2': dm_family('ctx2', True, easy=e), 'MultSen Dly DM': dm_family('multi', True, easy=e),
            'DMS': match_family(False, False, mo), 'DNMS': match_family(False, True, mo), 'DMC': match_family(True, False),
            'DNMC': match_family(True, True),
            'Go Mod1': go_family('fd', mod=0), 'Go Mod2': go_family('fd', mod=1), 'Sac': go_family('dly', mod=0),
            'Pursuit': pursuit(rb), 'Cat': two_afc('cat'), 'Match': two_afc('match'), 'Cat-B': two_afc('cat', rb),
            'T-maze Vis': tmaze('vis'), 'T-maze Aud': tmaze('aud'),
        }
        if layout == 'partial':
            g = float(pursuit_share); assert 0.0 <= g <= 1.0
            gens['Pursuit'] = pursuit(1, share=g)
        self.pursuit_share = pursuit_share
        self.names = list(tasks) if tasks is not None else YANG20 + ANCHORS + (MOUSE_TASKS if mouse_tasks else [])
        self.gens = [gens[t] for t in self.names]
        self.n_rules = len(self.names)                        # = number of TASKS (the task index is the trial's 'rule' everywhere)
        # rule input units: one per task, except that the T-maze tasks share one
        units, self.rule_map = {}, []
        for t in self.names:
            key = 'T-maze' if t in MOUSE_TASKS else t
            self.rule_map.append(units.setdefault(key, len(units)))
        self.n_rule_units = len(units)
        self.n_in_rings = N_IN_RINGS + (1 if any(t == 'T-maze Vis' for t in self.names) else 0)
        self.render_kw = {}
        if self.n_in_rings != N_IN_RINGS or self.n_rule_units != self.n_rules:
            self.render_kw = dict(n_in_rings=self.n_in_rings, rule_map=np.asarray(self.rule_map))
        self.n_inputs = 1 + self.n_in_rings * N_RING + self.n_rule_units
        self.n_outputs = 1 + N_OUT_RINGS * N_RING
        w = np.array([ctx_weight if t in ('Ctx DM 1', 'Ctx DM 2') else 1.0 for t in self.names])
        self.p = w / w.sum()                                  # Yang: Ctx DM 1 / 2 five times as frequent

    def index(self, name):
        return self.names.index(name)

    def sample(self, rng, n, task=None, test=False, tasks=None):
        """n trial parameter sets. task: one task index (or name) for all trials; tasks: an (n,) array of task indices; neither:
        tasks drawn with the training frequencies."""
        if isinstance(task, str):
            task = self.index(task)
        if tasks is None:
            tasks = np.full(n, task) if task is not None else rng.choice(self.n_rules, n, p=self.p)
        out = _new(n); out['rule'] = tasks.astype(int); metas = {}
        for k in np.unique(tasks):
            idx = np.flatnonzero(tasks == k)
            b = self.gens[k](rng, len(idx), test)
            for key, v in b.items():
                if key != 'meta':
                    out[key][idx] = v
            for key, v in b['meta'].items():
                metas.setdefault(key, np.full(n, np.nan))[idx] = v
        out['meta'] = metas
        return out


# ═══ rendering (torch) ══════════════════════════════════════════════════════════════════════════════════════════════════
def _bump(dirs, pref):
    d = torch.abs(dirs[..., None] - pref)
    d = torch.minimum(d, 2 * np.pi - d)
    return 0.8 * torch.exp(-0.5 * (8 * d / np.pi) ** 2)


def render(b, n_rules, device, sigma_in=0.01, alpha=0.2, noise=True, T_pad=None, rule_pattern=None, pre_on=0, n_in_rings=N_IN_RINGS,
           rule_map=None):
    """Trial parameters -> dict of torch tensors on `device`: inputs (n, T, I), targets (n, T, O), mask (n, T, O), valid (n, T),
    T (n,), resp_on (n,), respond (n,), resp_dir (n,), resp_ring (n,). pre_on: the first pre_on steps carry no loss (reference code:
    100 ms = 5 steps; Methods: not mentioned, default 0). Input noise sqrt(2 / alpha) sigma_in N(0, 1) on every
    input channel (Yang). rule_pattern (n, n_rules) replaces the one-hot rule (compositional rule inputs).
    n_in_rings / rule_map: Battery.render_kw (a 4th input ring; task index -> rule unit). resp_share > 0: that fraction of the response
    bump goes on ring A and the rest on resp_ring (layout 'partial')."""
    f = lambda k, dt=torch.float32: torch.as_tensor(np.asarray(b[k]), dtype=dt, device=device)
    n = len(b['T']); Tm = int(b['T'].max()) if T_pad is None else int(T_pad)
    t = torch.arange(Tm, device=device, dtype=torch.float32)
    T, fix_off, resp_on = f('T'), f('fix_off'), f('resp_on')
    valid = t[None] < T[:, None]
    fix_in = ((t[None] < fix_off[:, None]) & valid).float()
    on, off = f('ev_on'), f('ev_off')
    amp = torch.clamp(f('ev_s0')[..., None] + f('ev_slope')[..., None] * (t - on[..., None]), min=0.0)
    amp = amp * ((t >= on[..., None]) & (t < off[..., None])).float()                       # (n, E, T)
    pref = torch.as_tensor(PREF, dtype=torch.float32, device=device)
    bump = _bump(f('ev_dir'), pref)                                                          # (n, E, 32)
    ring = f('ev_ring', torch.long)
    rings = [torch.einsum('net,nei->nti', amp * (ring == r).float()[..., None], bump) for r in range(n_in_rings)]
    if rule_pattern is None and rule_map is not None:
        rm = np.asarray(rule_map)
        rule = torch.nn.functional.one_hot(torch.as_tensor(rm[np.asarray(b['rule'])], dtype=torch.long, device=device),
                                           int(rm.max()) + 1).float()
    elif rule_pattern is None:
        rule = torch.nn.functional.one_hot(f('rule', torch.long), n_rules).float()
    else:
        rule = torch.as_tensor(rule_pattern, dtype=torch.float32, device=device)
    rule = rule[:, None, :] * valid[..., None].float()
    u = torch.cat([fix_in[..., None], *rings, rule], -1)
    if noise and sigma_in > 0:
        u = u + np.sqrt(2.0 / alpha) * sigma_in * torch.randn_like(u)
    respond = f('respond', torch.bool)
    after = t[None] >= resp_on[:, None]
    go = after & respond[:, None]
    fix_t = torch.where(go, 0.05, 0.85)
    rb = _bump(f('resp_dir'), pref) + 0.05                                                   # (n, 32)
    rr = f('resp_ring', torch.long)
    outs = [fix_t[..., None]]
    share = np.asarray(b['resp_share'], float) if 'resp_share' in b else np.zeros(n)
    if (share > 0).any():                                                                    # layout 'partial'
        sh = torch.as_tensor(share, dtype=torch.float32, device=device)
        bp = rb - 0.05
        for k in range(N_OUT_RINGS):
            gain = torch.where(rr == k, 1.0 - sh, torch.zeros_like(sh)) + (torch.where(rr != 0, sh, torch.zeros_like(sh)) if k == 0 else 0.0)
            on_k = (go & (gain > 0)[:, None])[..., None]
            outs.append(torch.where(on_k, (gain[:, None] * bp + 0.05)[:, None, :], torch.full_like(rb[:, None, :], 0.05)))
    else:
        for k in range(N_OUT_RINGS):
            on_k = (go & (rr == k)[:, None])[..., None]
            outs.append(torch.where(on_k, rb[:, None, :], torch.full_like(rb[:, None, :], 0.05)))
    y = torch.cat(outs, -1)
    base = torch.where(t[None] < resp_on[:, None], 1.0, torch.where(t[None] < resp_on[:, None] + GRACE, 0.0, 5.0)) * valid.float()
    if pre_on:
        base = base * (t[None] >= pre_on).float()
    mask = torch.cat([2 * base[..., None], base[..., None].expand(n, Tm, N_OUT_RINGS * N_RING)], -1)
    return dict(inputs=u, targets=y, mask=mask, valid=valid, T=T.long(), resp_on=resp_on.long(), respond=respond,
                resp_dir=f('resp_dir'), resp_ring=rr)


# ═══ read-outs from output traces ═══════════════════════════════════════════════════════════════════════════════════════
def popvec(z_ring):
    """(..., 32) ring activity -> direction in [0, 2 pi)."""
    z = np.asarray(z_ring)
    return np.arctan2((z * np.sin(PREF)).sum(-1), (z * np.cos(PREF)).sum(-1)) % (2 * np.pi)


def angdist(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def ring_slice(k):
    return slice(1 + k * N_RING, 1 + (k + 1) * N_RING)


def performance(z_last, respond, resp_dir, resp_ring, resp_share=None):
    """Yang's rule on the output at the last valid step. z_last (n, O) numpy. Returns (n,) bool. resp_share > 0 (layout
    'partial'): the direction is read from the sum of ring A and the trial's own ring."""
    fix = z_last[:, 0]
    rr = np.asarray(resp_ring)
    ring = np.stack([z_last[i, ring_slice(rr[i])] for i in range(len(rr))])
    if resp_share is not None and (np.asarray(resp_share) > 0).any():
        sp = np.flatnonzero(np.asarray(resp_share) > 0)
        ring[sp] = ring[sp] + z_last[sp, ring_slice(0)]
    ok_loc = angdist(popvec(ring), resp_dir) < 0.2 * np.pi
    return np.where(respond, (fix < 0.5) & ok_loc, fix > 0.5)


def resp_ring_rates(Z, rr, share):
    """(..., O) output -> (..., 32) the response ring of a trial (resp_ring rr); with resp_share > 0 (layout 'partial') ring A is
    added, since the bump is split between the two rings in the same direction."""
    out = Z[..., ring_slice(int(rr))]
    return out + Z[..., ring_slice(0)] if (float(share) > 0 and int(rr) != 0) else out


def hemifield(direction):
    """1 = right (-90 .. +90 deg), 0 = left."""
    return (np.cos(direction) > 0).astype(int)
