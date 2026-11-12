"""Yang et al. (2019) leaky RNN, K independent networks stacked on a leading axis (every network keeps its own weights,
inputs and noise).

    r_t = (1 - alpha) r_{t-1} + alpha f(W_rec r_{t-1} + W_in u_t + b + sqrt(2 / alpha) sigma_rec xi_t),   alpha = dt / tau
    z_t = sigmoid(W_out r_t + b_out)

Note the RATE dynamics (Yang's form; the leak acts on r), not the current-based form in which the leak acts on x
with r = f(x). f: softplus (Yang's reference), relu, tanh, retanh = max(tanh, 0). Initialisation (Yang Methods): W_rec = 0.5 I ('diag')
or 0.5 x a random orthogonal matrix ('orth'); W_in ~ N(0, 1 / N_in); W_out ~ N(0, (0.4 / sqrt(N))^2); biases 0.
Silencing a unit (Yang's lesion: its outgoing recurrent and output weights set to 0) = multiplying its rate by 0 at every step.
Network k is initialised from its own seed only, so a network does not depend on which others share its batch.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACT = {'softplus': F.softplus, 'relu': F.relu, 'tanh': torch.tanh, 'retanh': lambda x: torch.clamp(torch.tanh(x), min=0.0)}


def init_weights(seed, n_in, n_out, N, init='diag', rec_gain=0.5):
    g = torch.Generator().manual_seed(int(seed))
    W_in = torch.randn(N, n_in, generator=g) / np.sqrt(n_in)
    if init == 'diag':
        W_rec = rec_gain * torch.eye(N)
    elif init == 'orth':
        q, r = torch.linalg.qr(torch.randn(N, N, generator=g))
        W_rec = rec_gain * q * torch.sign(torch.diagonal(r))[None, :]
    else:
        raise ValueError(init)
    W_out = torch.randn(n_out, N, generator=g) * (0.4 / np.sqrt(N))
    return W_in, W_rec, torch.zeros(N), W_out, torch.zeros(n_out)


class BatchedYangRNN(nn.Module):
    def __init__(self, n_inputs, n_outputs, seeds, n_units=256, tau_ms=100.0, dt_ms=20.0, act='softplus', init='diag',
                 rec_gain=0.5, sigma_rec=0.05):
        super().__init__()
        self.seeds = [int(s) for s in seeds]
        self.K, self.N = len(self.seeds), int(n_units)
        self.n_inputs, self.n_outputs = int(n_inputs), int(n_outputs)
        self.alpha = float(dt_ms) / float(tau_ms)
        self.act, self.sigma_rec = act, float(sigma_rec)
        self.kw = dict(n_units=n_units, tau_ms=tau_ms, dt_ms=dt_ms, act=act, init=init, rec_gain=rec_gain, sigma_rec=sigma_rec)
        ws = [init_weights(s, self.n_inputs, self.n_outputs, self.N, init, rec_gain) for s in self.seeds]
        for i, name in enumerate(('W_in', 'W_rec', 'b', 'W_out', 'b_out')):
            setattr(self, name, nn.Parameter(torch.stack([w[i] for w in ws])))

    def f(self, x):
        return ACT[self.act](x)

    def clip_per_network(self, clip):
        """Scale every network's gradient to norm <= clip (independently, as clip_grad_norm_ would do per network).
        Not used by train.py, which clips element-wise as Yang's reference code does."""
        ps = [p for p in self.parameters() if p.grad is not None]
        norms = torch.sqrt(sum((p.grad.flatten(1) ** 2).sum(1) for p in ps))
        coef = torch.clamp(clip / (norms + 1e-6), max=1.0)
        for p in ps:
            p.grad.mul_(coef.view(-1, *([1] * (p.grad.ndim - 1))))
        return norms

    def forward(self, u, noise=True, silence=None, keep_rates=True, r0=None, w_rec_mask=None):
        """u (K, B, T, I) -> rates (K, B, T, N) [None if keep_rates is False], outputs z (K, B, T, O).
        silence (K, N) or (N,) 0/1 rate mask; w_rec_mask (K, N, N) multiplies W_rec (connection perturbations)."""
        K, B, T, _ = u.shape
        a = self.alpha
        sd = np.sqrt(2.0 / a) * self.sigma_rec
        inp = torch.einsum('kbti,kni->kbtn', u, self.W_in) + self.b[:, None, None, :]
        W = self.W_rec if w_rec_mask is None else self.W_rec * w_rec_mask
        WrT = W.transpose(1, 2)
        WoT = self.W_out.transpose(1, 2)
        if silence is not None:
            silence = torch.as_tensor(silence, dtype=u.dtype, device=u.device)
            silence = silence.reshape(1 if silence.ndim == 1 else K, 1, self.N).expand(K, 1, self.N)
        r = torch.zeros(K, B, self.N, device=u.device, dtype=u.dtype) if r0 is None else r0
        rates, zs = [], []
        for t in range(T):
            pre = torch.baddbmm(inp[:, :, t], r, WrT)
            if noise and self.sigma_rec > 0:
                pre = pre + sd * torch.randn_like(pre)
            r = (1 - a) * r + a * self.f(pre)
            if silence is not None:
                r = r * silence
            if keep_rates:
                rates.append(r)
            else:
                zs.append(torch.sigmoid(torch.baddbmm(self.b_out[:, None, :], r, WoT)))
        if keep_rates:
            R = torch.stack(rates, 2)
            Z = torch.sigmoid(torch.einsum('kbtn,kon->kbto', R, self.W_out) + self.b_out[:, None, None, :])
            return R, Z
        return None, torch.stack(zs, 2)

    def extract(self, k):
        """Network k alone (K = 1), weights copied."""
        net = BatchedYangRNN(self.n_inputs, self.n_outputs, [self.seeds[k]], **self.kw)
        with torch.no_grad():
            for name in ('W_in', 'W_rec', 'b', 'W_out', 'b_out'):
                getattr(net, name).copy_(getattr(self, name)[k:k + 1].detach().cpu())
        return net

    @classmethod
    def stack(cls, nets):
        """Several K = 1 networks (same hyperparameters) -> one batched module."""
        kw = nets[0].kw
        out = cls(nets[0].n_inputs, nets[0].n_outputs, [n.seeds[0] for n in nets], **kw)
        with torch.no_grad():
            for name in ('W_in', 'W_rec', 'b', 'W_out', 'b_out'):
                getattr(out, name).copy_(torch.cat([getattr(n, name).detach().cpu() for n in nets]))
        return out
