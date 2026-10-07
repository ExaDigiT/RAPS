"""
ML surrogate for the FMU cooling model.

`SurrogateCoolingModel` is a drop-in replacement for `ThermoFluidsModel`: the
engine calls `simulate_cooling()` at every power update and gets back the same
(cooling_inputs, cooling_outputs) dicts keyed by FMU variable names. Instead of
integrating the Modelica model it runs a trained multi-rate network (the
grouped DeepM&Mnet of the hpc-cooling-surrogate project) exported as a bundle:

    <surrogate_path>/model.ts      TorchScript: (u_fast, y_fast, u_slow, y_slow)
                                   -> one normalized-delta tensor per output group
    <surrogate_path>/bundle.json   columns, rates, histories, normalization,
                                   training input ranges, start-up initializer

How it steps
------------
The network predicts the next `stride_s` (30 s) of every CDU output from the
recent history of inputs and outputs, sampled as block means on a fast (3 s)
and a slow (30 s) grid. The model keeps those histories itself and runs closed
loop: its own predictions become the output history for the next stride. RAPS
calls in every engine tick; each call records the inputs as held over
[t, t + time_delta) and returns the outputs at t + time_delta, running one
forward pass per stride. (The FMU path instead advances the FMU by
POWER_UPDATE_FREQ per call whatever the tick length; the surrogate follows
simulation time.) Like the FMU, an output at t + dt only uses inputs up
to the start of its stride.

Start-up
--------
There is no history at the first call. The output history is filled from a
quasi-equilibrium fit in the bundle (outputs as a quadratic in each CDU's heat
load and the wet-bulb temperature), and then the network is relaxed for
`SURROGATE_WARMUP_S` (default 1200 s) of model time with the first inputs held
constant, so the first reported state is the network's own near-equilibrium.

Scope
-----
The surrogate predicts the 11 per-CDU outputs (temperatures, pressures, flows,
CDU pump power) in the units RAPS displays (degC, psig, GPM, kW). It does not
model the central energy plant, so the plant pump and cooling-tower powers are
absent and PUE only counts IT power and CDU pumps.
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np
from uncertainties import unumpy

from raps.cooling import ThermoFluidsModel


class SurrogateCoolingModel(ThermoFluidsModel):
    """Cooling model backed by a trained surrogate bundle instead of an FMU."""

    def __init__(self, **config):
        super().__init__(**config)
        self.bundle = None
        self.model = None
        self._started = False

    # ── setup ────────────────────────────────────────────────────────────────
    def initialize(self):
        import torch

        path = self.config.get('SURROGATE_PATH')
        if not path:
            raise ValueError("cooling_model='surrogate' needs cooling.surrogate_path "
                             "in the system config (or --system.cooling.surrogate-path)")
        path = Path(path)
        print(f'Initializing cooling surrogate from {path} ...')
        b = json.loads((path / 'bundle.json').read_text())
        if b.get('format_version') != 1:
            raise ValueError(f"unsupported surrogate bundle format {b.get('format_version')}")
        if b['num_cdus'] != self.model_num_cdus:
            raise ValueError(f"surrogate has {b['num_cdus']} CDUs, but the cooling config "
                             f"expects {self.model_num_cdus} (set cooling.fmu_num_cdus when "
                             f"borrowing a larger system's model)")
        self.bundle = b
        self._torch = torch
        self.model = torch.jit.load(str(path / 'model.ts'), map_location='cpu').eval()

        self.input_cols = b['input_cols']
        self.dyn_cols = b['dynamic_cols']
        self.n_in, self.n_dyn = len(self.input_cols), len(self.dyn_cols)
        self.stride = int(b['stride_s'])
        self.fast, self.slow = b['branches']['fast'], b['branches']['slow']
        self.heads = b['heads']
        self.u_mean = np.asarray(b['input_norm']['mean'], np.float32)
        self.u_std = np.asarray(b['input_norm']['std'], np.float32)
        self.y_mean = np.asarray(b['output_norm']['mean'], np.float32)
        self.y_std = np.asarray(b['output_norm']['std'], np.float32)
        self.dscale = {int(r): np.asarray(v, np.float32) for r, v in b['delta_scale'].items()}
        self.u_lo = np.asarray(b['input_range']['min'], np.float32)
        self.u_hi = np.asarray(b['input_range']['max'], np.float32)
        self.warmup_s = int(self.config.get('SURROGATE_WARMUP_S', 1200))

        # Input columns are FMU input names, matched against generate_runtime_values().
        missing = [c for c in self.input_cols if not self._is_runtime_key(c)]
        if missing:
            raise ValueError(f"surrogate inputs not produced by RAPS: {missing[:3]}")

        # Which input feeds each output's start-up fit: its own CDU's Q_flow.
        q_idx = {i: self.input_cols.index(
            f"simulator_1_datacenter_1_computeBlock_{i + 1}_cabinet_1_sources_Q_flow_total")
            for i in range(b['num_cdus'])}
        cdu = [int(c.split('computeBlock[')[1].split(']')[0]) - 1 for c in self.dyn_cols]
        self._ss_q = np.asarray([q_idx[c] for c in cdu])
        self._ss_w = next(i for i, c in enumerate(self.input_cols) if 'Q_flow' not in c)
        self._ss_coef = np.asarray(b['steady_state']['coef'], np.float64)

        self.inputs, self.outputs = [], []        # no FMU variables
        self._warned_range = False
        self.extrapolating = False  # inputs outside the training range on the latest call

    def _is_runtime_key(self, col):
        return (col.endswith('_sources_Q_flow_total')
                or col in self.config['TEMPERATURE_KEYS'])

    # ── numerics ─────────────────────────────────────────────────────────────
    def _steady_state(self, u):
        q = u[self._ss_q] / 1e6
        w = u[self._ss_w] - 290.0
        X = np.stack([np.ones_like(q), q, w * np.ones_like(q), q * q, q * w,
                      w * w * np.ones_like(q)], axis=1)
        return np.einsum('ij,ij->i', X, self._ss_coef).astype(np.float32)

    def _u_blocks(self, origin, rate, history):
        """Block means of the 1 s input record over [origin - history*rate, origin)."""
        i0 = origin - history * rate - self._u_t0
        seg = self._u_rec[i0:i0 + history * rate]
        return seg.reshape(history, rate, self.n_in).mean(axis=1)

    def _forward(self, origin):
        """One stride from `origin`: returns (new_fast (k_f, N), new_slow (N,))."""
        torch = self._torch
        uf = (self._u_blocks(origin, self.fast['rate'], self.fast['history']) - self.u_mean) / self.u_std
        us = (self._u_blocks(origin, self.slow['rate'], self.slow['history']) - self.u_mean) / self.u_std
        yf = (self._y_fast - self.y_mean) / self.y_std
        ys = (self._y_slow - self.y_mean) / self.y_std
        with torch.no_grad():
            outs = self.model(*(torch.from_numpy(np.ascontiguousarray(a[None], dtype=np.float32))
                                for a in (uf, yf, us, ys)))
        k_fast = self.stride // self.fast['rate']
        new_fast = np.empty((k_fast, self.n_dyn), np.float32)
        new_slow = np.empty(self.n_dyn, np.float32)
        for head, out in zip(self.heads, outs):
            idx, rate = head['indices'], head['rate']
            k = self.stride // rate
            d = out[0, :k].numpy() * self.dscale[rate][idx]
            if head['branch'] == 'fast':
                absd = self._y_fast[-1, idx] + np.cumsum(d, axis=0)      # (k_fast, n)
                new_fast[:, idx] = absd
                new_slow[idx] = absd.mean(axis=0)                         # store's decimation
            else:
                nxt = self._y_slow[-1, idx] + d[0]
                new_slow[idx] = nxt
                prev = self._y_fast[-1, idx]
                w = (np.arange(1, k_fast + 1) / k_fast)[:, None]
                new_fast[:, idx] = prev * (1 - w) + nxt * w               # slow -> fast
        return new_fast, new_slow

    def _commit(self, new_fast, new_slow):
        self._y_fast = np.concatenate([self._y_fast[len(new_fast):], new_fast])
        self._y_slow = np.concatenate([self._y_slow[1:], new_slow[None]])
        self._origin += self.stride
        self._pending = None

    def _record_inputs(self, t, dt, u):
        """Record inputs held over [t, t + dt) in the 1 s record, trimming old data.

        A gap since the previous call is filled with the previous inputs (they
        were held until now); an overlap is overwritten from t onward.
        """
        end = self._u_t0 + len(self._u_rec)
        if t > end:
            self._u_rec = np.concatenate([self._u_rec, np.repeat(self._u_rec[-1:], t - end, axis=0)])
        elif t < end:
            self._u_rec = self._u_rec[:max(t - self._u_t0, 0)]
        self._u_rec = np.concatenate([self._u_rec, np.repeat(u[None], dt, axis=0)])

    def _trim_inputs(self):
        """Drop input history older than the next forward pass can need."""
        span = max(self.fast['history'] * self.fast['rate'],
                   self.slow['history'] * self.slow['rate'])
        drop = (self._origin - span) - self._u_t0
        if drop > 0:
            self._u_rec = self._u_rec[drop:]
            self._u_t0 += drop

    def _start(self, t, u):
        """Fill histories at the first call, then relax under constant inputs."""
        span = max(self.fast['history'] * self.fast['rate'],
                   self.slow['history'] * self.slow['rate'])
        self._u_t0 = t - span - self.warmup_s
        self._u_rec = np.repeat(u[None], span + self.warmup_s, axis=0)
        y0 = self._steady_state(u)
        self._y_fast = np.repeat(y0[None], self.fast['history'], axis=0)
        self._y_slow = np.repeat(y0[None], self.slow['history'], axis=0)
        self._origin = t - self.warmup_s
        self._pending = None
        while self._origin < t:
            self._commit(*self._forward(self._origin))
        self._started = True

    def _advance(self, t, dt, u):
        """Record inputs for [t, t + dt) and return the outputs at t + dt."""
        if not self._started:
            self._start(t, u)
        self._record_inputs(t, dt, u)
        target = t + dt
        while self._origin + self.stride <= target:
            fwd = self._pending if self._pending is not None else self._forward(self._origin)
            self._commit(*fwd)
        self._trim_inputs()
        offset = target - self._origin
        if offset == 0:
            return self._y_fast[-1].copy()
        if self._pending is None:
            self._pending = self._forward(self._origin)
        new_fast = self._pending[0]
        k = min(int(np.ceil(offset / self.fast['rate'])) - 1, len(new_fast) - 1)
        return new_fast[k].copy()

    # ── RAPS interface ───────────────────────────────────────────────────────
    def simulate_cooling(self, *, rack_power, engine):
        cdu_power = rack_power.T[-1] * 1000
        runtime_values = self.generate_runtime_values(cdu_power, engine)
        u = np.asarray([float(unumpy.nominal_values(runtime_values[c])) for c in self.input_cols],
                       np.float32)
        self.extrapolating = bool(((u < self.u_lo) | (u > self.u_hi)).any())  # read by the UI
        if not self._warned_range and self.extrapolating:
            bad = [c for c, x, lo, hi in zip(self.input_cols, u, self.u_lo, self.u_hi)
                   if x < lo or x > hi]
            warnings.warn(f"cooling surrogate: {len(bad)} inputs outside the training range "
                          f"(e.g. {bad[0]}); they are clipped to it. Further warnings suppressed.")
            self._warned_range = True
        u = np.clip(u, self.u_lo, self.u_hi)

        t = int(engine.current_timestep)
        dt = max(int(getattr(engine, 'time_delta', 1) or 1), 1)
        y = self._advance(t, dt, u)

        cooling_inputs = dict(zip(self.input_cols, u.tolist()))
        cooling_outputs = self.to_system_frame(dict(zip(self.dyn_cols, y.tolist())))
        cooling_outputs['pue'] = self.calculate_pue(
            self.pue_inputs({k: v for k, v in cooling_inputs.items() if 'Q_flow' in k}),
            cooling_outputs)
        cooling_inputs['time'] = t
        self.fmu_history.append({**cooling_inputs, **cooling_outputs})
        return cooling_inputs, cooling_outputs

    def terminate(self):
        pass

    def cleanup(self):
        pass
