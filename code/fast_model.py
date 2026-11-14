"""
Fast copy of the reactor equations for the safety brake's look-ahead.

The brake simulates the reactor hundreds of times per decision, so it spends almost all
of the AI training time inside the integrator. This module does exactly the same RK4
arithmetic as reactor_model.advance(), but on plain Python floats instead of small
NumPy arrays (NumPy is slow for 5-element vectors). Same equations, same step size,
same order of operations. test_fast_model() checks the two agree to 1e-9 relative.

reactor_model.py stays the single, verified reference. Only shield.py uses this file.
"""
from __future__ import annotations

import math

import numpy as np

from reactor_model import R_GAS, ReactorParams


class FastModel:
    """Pre-computed constants for one parameter set (one 'plant' or one shield model)."""

    def __init__(self, p: ReactorParams):
        if p.water_order != 0.0:
            raise ValueError("FastModel supports the default pseudo-first-order rate law only")
        self.p = p
        self.k_ref, self.Ea_R, self.T_ref = p.k_ref, p.Ea / R_GAS, p.T_ref
        self.molA = p.mol_per_mL_A / 60.0          # (mol/s) per (mL/min)
        self.Cp_A, self.Cp_W, self.Cp_P = p.Cp_A, p.Cp_W, p.Cp_P
        self.mdH, self.dH_mix = -p.dH_r, p.dH_mix
        self.UA, self.Tj, self.Tf = p.UA, p.T_jacket, p.T_feed
        self.dCp_end = 2 * p.Cp_P - p.Cp_W - p.Cp_A   # used by MTSR

    # same expression order as reactor_model.rhs
    def rhs(self, nA, nW, nP, T, F):
        F_mol = F / 60.0 * self.p.mol_per_mL_A
        r = self.k_ref * math.exp(-self.Ea_R * (1.0 / T - 1.0 / self.T_ref)) * (nA if nA > 0.0 else 0.0)
        Q = (F_mol * self.Cp_A * (self.Tf - T) + self.mdH * r + (-F_mol * self.dH_mix)
             - self.UA * (T - self.Tj))
        C = nA * self.Cp_A + nW * self.Cp_W + nP * self.Cp_P
        return F_mol - r, -r, 2.0 * r, Q / C, F_mol

    def advance(self, s, F, dt, substeps=10):
        """s = (nA, nW, nP, T, nFed) tuple -> tuple after dt seconds at constant feed F (mL/min)."""
        h = dt / substeps
        a, w, pp, T, fd = s
        rhs = self.rhs
        for _ in range(substeps):
            k1 = rhs(a, w, pp, T, F)
            k2 = rhs(a + 0.5 * h * k1[0], w + 0.5 * h * k1[1], pp + 0.5 * h * k1[2], T + 0.5 * h * k1[3], F)
            k3 = rhs(a + 0.5 * h * k2[0], w + 0.5 * h * k2[1], pp + 0.5 * h * k2[2], T + 0.5 * h * k2[3], F)
            k4 = rhs(a + h * k3[0], w + h * k3[1], pp + h * k3[2], T + h * k3[3], F)
            c = h / 6.0
            a = a + c * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            w = w + c * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            pp = pp + c * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            T = T + c * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
            fd = fd + c * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4])
        return (a, w, pp, T, fd)

    def mtsr(self, s):
        """Same formula as reactor_model.mtsr (Stoessel MTSR)."""
        a, w, pp, T, _ = s
        nA = a if a > 0.0 else 0.0
        C = a * self.Cp_A + w * self.Cp_W + pp * self.Cp_P
        C_end = C - nA * self.Cp_A + nA * (2 * self.Cp_P - self.Cp_W)
        return T + nA * self.mdH / C_end


def test_fast_model():
    """Fast and reference integrators agree on a full constant-feed batch and on F = 0 decay."""
    from reactor_model import advance, initial_state, mtsr
    for p in (ReactorParams(), ReactorParams().with_mismatch(1.2, 1.2, 0.8),
              ReactorParams().with_mismatch(0.8, 0.8, 1.2)):
        fm = FastModel(p)
        x = initial_state(p); s = tuple(x)
        for i in range(400):
            F = 20.0 if i < 60 else (0.0 if i < 120 else 7.3)
            x = advance(x, F, p, 10.0); s = fm.advance(s, F, 10.0)
            assert np.allclose(x, s, rtol=1e-9, atol=1e-12), (i, x, s)
            assert abs(mtsr(x, p) - fm.mtsr(s)) < 1e-9
            x4 = advance(x, 0.0, p, 10.0, substeps=4); s4 = fm.advance(s, 0.0, 10.0, substeps=4)
            assert np.allclose(x4, s4, rtol=1e-9, atol=1e-12)
    print("fast_model: identical to reactor_model to 1e-9")


if __name__ == "__main__":
    test_fast_model()
