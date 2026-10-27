"""Finite artificial chemistry, exact aggregation and scoped diagnostics.

All symbols and feature distances are dimensionless digital quantities.
Row-vector convention: an unnormalised live distribution evolves as p @ M.
The two-width extension is a declared diagnostic control, not a refit.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from decimal import Decimal, localcontext
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.optimize import brentq

def parameters(sigma_r, sigma_d=None):
    sigma_d = sigma_r if sigma_d is None else sigma_d
    if any(not math.isfinite(s) or s < 0 for s in (sigma_r, sigma_d)):
        raise ValueError("Widths must be finite and nonnegative")
    u = math.exp(-1 / sigma_r**2) if sigma_r else 0.0
    td = math.exp(-1 / (2 * sigma_d**2)) if sigma_d else 0.0
    return u, 2 * td / (1 + td)**2

def reader_classes(states):
    states = np.asarray(states)
    if states.shape[1] != 4:
        raise ValueError("The Z,C,A,P aggregation is specific to m=2")
    return (states[:, 0] != states[:, 2]).astype(int) + 2 * (states[:, 1] != states[:, 3]).astype(int)

def lumped_matrix(sigma_r, sigma_d=None):
    u, a = parameters(sigma_r, sigma_d)
    return np.array([[u, 0, 0, 0], [1, 0, 0, 0], [u/4]*4,
                     [a*a, a*(1-a), a*(1-a), (1-a)**2]], dtype=float)

def channels(sigma_r, sigma_d=None, g=(0, 0, 1, 1)):
    sigma_d = sigma_r if sigma_d is None else sigma_d
    parameters(sigma_r, sigma_d)
    g = tuple(g); q = max(g) + 1
    if len(g) % 2 or set(g) != set(range(q)):
        raise ValueError("Use an even word containing every standard input ID")
    states = np.array(list(itertools.product(range(q), repeat=len(g))), dtype=int)
    keys, values = states[:, ::2], states[:, 1::2]
    distances = (np.arange(q)[None, :, None] - keys[:, None, :])**2
    minimum = distances.min(axis=2)
    gate = (minimum == 0).astype(float) if sigma_r == 0 else np.exp(-minimum/(2*sigma_r**2))
    if sigma_d == 0:
        # Nearest-slot limit; unmatched inputs fail under the original zero
        # gate. Off-diagonal zero-width controls are not used in this study.
        weights = (distances == minimum[:, :, None]).astype(float)
    else:
        weights = np.exp(-(distances-minimum[:, :, None])/(2*sigma_d**2))
        if np.any(weights == 0): raise ArithmeticError("Positive conditional weights underflowed")
    weights /= weights.sum(axis=2, keepdims=True)
    output = np.zeros((len(states), q, q))
    for value in range(q):
        output[:, :, value] = (weights*(values[:, None, :] == value)).sum(axis=2)
    return states, gate, output

def build_matrix(sigma_r, sigma_d=None, g=(0, 0, 1, 1)):
    states, gate, output = channels(sigma_r, sigma_d, g)
    c = np.prod(gate[:, list(g)], axis=1)
    M = np.broadcast_to(c[:, None], (len(states), len(states))).copy()
    for j, symbol in enumerate(g): M *= output[:, symbol, states[:, j]]
    return M, states

def native_mask(states, g=(0, 0, 1, 1)):
    reference = Counter(zip(g[::2], g[1::2]))
    return np.array([Counter(zip(p[::2], p[1::2])) == reference for p in states])

def lifetime_closed(sigma_r, sigma_d=None):
    u, a = parameters(sigma_r, sigma_d)
    if a == 0 or u == 1: return math.inf
    return (4-5*u+u*u+(12-7*u)*a+(6*u-8)*a*a) / ((1-u)*a*(8-4*a-3*u+2*a*u))

def productive_eigenvalue(u, a):
    x, w = u/4, (1-a)**2
    return ((x+w) + math.sqrt((x-w)**2 + u*a*(1-a)))/2

def spectral_quantities(sigma_r, sigma_d=None):
    u, a = parameters(sigma_r, sigma_d)
    lam = productive_eigenvalue(u, a); rho = max(u, lam)
    # Full M has two homogeneous classes. No unique global eigenvector is
    # asserted when those classes dominate.
    if lam > u + 2e-14:
        va = a*(1-a)/(lam-u/4)
        vz = (va*(1+u/4) + a*a)/(lam-u)
        v = np.array([vz, va, va, 1.0]); v /= v.sum()
        native = (u*v[2]/8 + (1-2*a+a*a/2)*v[3])/lam
        regime = "repertoire_dominated"
    else:
        v = np.array([1., 0., 0., 0.]); native = 0.0
        regime = "homogeneous_dominated_or_boundary"
    return dict(u=u, a=a, lambda_repertoire=lam, rho=rho,
                death_threshold=rho/(1+rho), native_spectral_fraction=native,
                P_spectral_fraction=float(v[3]), regime=regime)

def decoding_information_p(sigma_d):
    if sigma_d == 0: return 1.0
    td = math.exp(-1/(2*sigma_d*sigma_d)); beta = td/(1+td)
    if beta == 0: return 1.0
    entropy = -(beta*math.log2(beta) + (1-beta)*math.log1p(-beta)/math.log(2))
    return 1-entropy

def curve_row(sigma_r, sigma_d=None, horizon=32):
    if not isinstance(horizon, int) or horizon < 0:
        raise ValueError("Horizon must be a nonnegative integer")
    sigma_d = sigma_r if sigma_d is None else sigma_d
    L = lumped_matrix(sigma_r, sigma_d)
    distribution = np.array([0., 0., 0., 1.]); previous = distribution.copy()
    for _ in range(horizon):
        previous = distribution.copy(); distribution = distribution @ L
    s = float(distribution.sum()); u, a = parameters(sigma_r, sigma_d)
    nr = 1.0 if horizon == 0 else float(previous[2]*u/8 + previous[3]*(1-2*a+a*a/2))
    mono = 0.0 if horizon == 0 else float(previous[0]*u + previous[1] + previous[2]*u/8 + previous[3]*a*a/2)
    return dict(sigma_r=float(sigma_r), sigma_d=float(sigma_d), h=horizon, C=1.0,
                S_h=s, native_joint=nr, native_conditional=nr/s,
                homogeneous_joint=mono, homogeneous_conditional=mono/s,
                two_value_joint=float(distribution[2]+distribution[3]),
                decoding_information_P_bits=decoding_information_p(sigma_d),
                decoding_information_joint_bits=float(distribution[3])*decoding_information_p(sigma_d),
                expected_trials=lifetime_closed(sigma_r, sigma_d), **spectral_quantities(sigma_r, sigma_d))

def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def decimal_lifetime(sigma, precision=90):
    with localcontext() as ctx:
        ctx.prec = precision
        s = Decimal(str(sigma)); t = (-1/(2*s*s)).exp(); u = t*t; a = 2*t/(1+t)**2
        tau = (4-5*u+u*u+(12-7*u)*a+(6*u-8)*a*a)/((1-u)*a*(8-4*a-3*u+2*a*u))
        return {'sigma':str(sigma), 'expected_trials':str(tau),
                'scaled_log_lifetime':str(2*s*s*tau.ln()), 'precision':precision}

def run(root):
    started = time.perf_counter(); out = Path(root)/'results/finite'; out.mkdir(parents=True, exist_ok=True)
    from analysis.crossing import run as crossing_run
    crossing_run(root)
    from analysis.critical import run as critical_run
    critical_run(root)
    sigma = np.unique(np.r_[0., np.geomspace(.15, 8, 181), .2, .5, 1., 2., 4.])
    rows = [curve_row(s) for s in sigma]; write_csv(out/'sigma_scan.csv', rows)
    widths = np.geomspace(.2, 4, 81)
    write_csv(out/'width_grid.csv', (curve_row(r,d) for r in widths for d in widths))
    write_csv(out/'low_noise_exact.csv', [decimal_lifetime(s) for s in [.1,.15,.2,.25]])
    checks=[]
    for target, g in [('identity',(0,0,1,1)),('swap',(1,0,0,1))]:
        for s in [0,.2,.5,1,2,4]:
            M, states=build_matrix(s,g=g); classes=reader_classes(states); L=lumped_matrix(s)
            error=max(np.max(np.abs(M[:,classes==c].sum(axis=1)-L[classes,c])) for c in range(4))
            initial=np.flatnonzero(np.all(states==g,axis=1))[0]; v=np.zeros(16); v[initial]=1
            for _ in range(32): v=v@M
            full_native=float(v[native_mask(states,g)].sum()); row=curve_row(s)
            checks.append(dict(target=target,sigma=s,lumping_max_error=float(error),survival_error=abs(v.sum()-row['S_h']),native_reward_error=abs(full_native-row['native_joint'])))
            np.savez_compressed(out/f'{target}_sigma_{s:g}.npz',M=M,states=states,g=np.array(g),native_mask=native_mask(states,g))
    write_csv(out/'aggregation_checks.csv',checks)
    crossing=brentq(lambda s: spectral_quantities(s)['lambda_repertoire']-spectral_quantities(s)['u'],.7,1.3)
    q3rows=[]; g=(1,0,0,1,2,2)
    for s in [0,.2,.5,1,2]:
        M,states=build_matrix(s,g=g); _,gate,_=channels(s,g=g); C=np.prod(gate[:,list(g)],axis=1)
        native=native_mask(states,g); starts=[('native',g),('non_native_cycle',(0,1,1,2,2,0))]
        v=np.zeros((2,len(states)))
        for i,(_,p) in enumerate(starts):v[i,np.flatnonzero(np.all(states==p,axis=1))[0]]=1
        for h in range(1,33):
            v=v@M
            if h in [1,2,4,8,16,32]:
                for i,(name,p) in enumerate(starts):
                    s_live=float(v[i].sum()); nr=float(v[i,native].sum())
                    q3rows.append(dict(sigma=s,start=name,h=h,states=729,S_h=s_live,native_joint=nr,native_conditional=nr/s_live if s_live else 0.,max_row_sum_error=float(np.max(np.abs(M.sum(axis=1)-C)))))
        np.savez_compressed(out/f'q3_transposition_sigma_{s:g}.npz',M=M,states=states,g=np.array(g),native_mask=native)
    write_csv(out/'q3_extension.csv',q3rows)
    summary=dict(status='complete',wall_seconds=time.perf_counter()-started,diagonal_points=len(rows),width_grid_points=len(widths)**2,diagonal_crossover_sigma=crossing,max_lumping_error=max(r['lumping_max_error'] for r in checks),max_native_reward_error=max(r['native_reward_error'] for r in checks),representative=[curve_row(s) for s in [.2,.5,1,2,4]],q3_states=729,q3_widths=[0,.2,.5,1,2],scope='Fixed standard features; information conditional on a prescribed uniform input prior; no biological calibration.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps({k:v for k,v in summary.items() if k!='representative'},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); run(parser.parse_args().root)
