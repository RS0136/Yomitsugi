"""Shared artificial chemistry. Observation labels never enter this module."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from bisect import bisect_right
import math
import numpy as np

_MIN_RANDOM = float(np.nextafter(0., 1.))

class NumericalPrecisionError(ArithmeticError):
    """A positive probability cannot be represented safely in float64."""

@dataclass(frozen=True)
class Chemistry:
    backbone_features: dict[int, float]
    symbol_features: dict[int, float]
    residue_key_features: dict[int, float]
    residue_backbone_features: dict[int, float]

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            table = {int(k): float(v) for k, v in getattr(self, name).items()}
            if not table or not all(math.isfinite(v) for v in table.values()):
                raise ValueError(f"{name}: nonempty finite scalar feature table required")
            object.__setattr__(self, name, table)
        if self.residue_key_features.keys() != self.residue_backbone_features.keys():
            raise ValueError("Residue key/backbone catalogs differ")

    def to_dict(self):
        return {name: dict(getattr(self, name)) for name in self.__dataclass_fields__}

def standard_chemistry(q=2):
    return Chemistry({0: 0.0}, {i:float(i) for i in range(q)},
                     {i:float(i) for i in range(q)}, {i:0.0 for i in range(q)})

class Kernel:
    def __init__(self, chemistry: Chemistry, sigma: float):
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and nonnegative")
        self.chemistry = Chemistry(**chemistry.to_dict())
        self.sigma = float(sigma)
        self.residues = tuple(sorted(self.chemistry.residue_key_features))
        self._indices = {a:i for i,a in enumerate(self.residues)}
        self.distribution = lru_cache(maxsize=65536)(self._distribution)
        self._word_table = lru_cache(maxsize=16384)(self._make_word_table)
        self._sampling_table = lru_cache(maxsize=16384)(self._make_sampling_table)
        self._full_table = lru_cache(maxsize=65536)(self._make_full_distribution)
        groups = {}
        for i,a in enumerate(self.residues):
            key = (self.chemistry.residue_key_features[a],self.chemistry.residue_backbone_features[a])
            groups.setdefault(key,[]).append(i)
        self._feature_groups = tuple(groups.values())

    def _distribution(self, p: tuple[int,...], b: int, c: int):
        if not p or len(p) % 2:
            raise ValueError("Reader must have a positive even length")
        ch = self.chemistry
        if any(a not in self._indices for a in p):
            raise ValueError("Unknown residue")
        distances = np.array([(ch.symbol_features[c]-ch.residue_key_features[k])**2
                              +(ch.backbone_features[b]-ch.residue_backbone_features[k])**2
                              for k in p[::2]], dtype=float)
        probs = np.zeros(len(self.residues))
        if self.sigma == 0:
            weights = (distances == 0).astype(float)
            if not weights.any():
                probs.setflags(write=False)
                return -math.inf, probs
            log_s = 0.0
        else:
            logs = -distances/(2*self.sigma**2)
            if not np.isfinite(logs).all():
                raise NumericalPrecisionError("Nonfinite log weight; high precision required")
            log_s = float(logs.max())
            weights = np.exp(logs-log_s)
            if np.any(weights == 0):
                raise NumericalPrecisionError("Positive conditional weight underflow; high precision required")
        weights /= weights.sum()  # shifted log-sum-exp
        for value, weight in zip(p[1::2], weights):
            probs[self._indices[value]] += weight
        probs /= probs.sum()
        probs.setflags(write=False)
        return log_s, probs

    def _make_word_table(self, p, b, g):
        if not g or len(g)%2:
            raise ValueError("Template must have a positive even length")
        rows = [self.distribution(p,b,c) for c in g]
        cumulative = np.array([np.cumsum(row[1]) for row in rows])
        cumulative.setflags(write=False)
        return math.fsum(row[0] for row in rows), cumulative

    def _make_sampling_table(self, p, b, g):
        log_c, cumulative = self._word_table(p,b,g)
        rows = tuple(tuple(row) for row in cumulative.tolist())
        # Exact single-output laws only: never drop positive small probabilities.
        conditional = [self.distribution(p,b,c)[1] for c in g]
        fixed = tuple(self.residues[int(np.argmax(prob))] for prob in conditional) if all(np.count_nonzero(prob)==1 for prob in conditional) else None
        return log_c, rows, fixed

    def log_success(self, p, b, g):
        return self._word_table(tuple(p),b,tuple(g))[0]

    def birth_probability(self, p, b, g):
        log_c = self.log_success(p,b,g)
        result = math.exp(log_c)
        if result == 0 and math.isfinite(log_c):
            raise NumericalPrecisionError("Positive word probability underflow; retain log_C")
        return result

    def read_word(self, p, b, g, rng):
        """Same uniform stream and boundaries as the vector reference, without tiny arrays."""
        log_c, cumulative, fixed = self._sampling_table(tuple(p),b,tuple(g))
        if log_c == -math.inf:
            return None
        if log_c < 0 and math.log(max(float(rng.random()),_MIN_RANDOM)) >= log_c:
            return None
        uniforms = rng.random(len(g))
        if fixed is not None:
            return fixed  # Uniforms deliberately consumed even for deterministic output.
        values = self.residues
        if len(values)==2:
            return tuple(values[u>=row[0]] for row,u in zip(cumulative,uniforms.tolist()))
        last = len(values)-1
        # Limiting the search to the first A-1 boundaries exactly reproduces
        # the reference's clipping of the last floating-point CDF endpoint.
        return tuple(values[bisect_right(row,u,hi=last)] for row,u in zip(cumulative,uniforms.tolist()))

    def full_distribution(self,p,b,c):
        return self._full_table(tuple(p),b,c)

    def _make_full_distribution(self,p,b,c):
        log_s, probs = self.distribution(tuple(p),b,c)
        success = math.exp(log_s)
        if success == 0 and math.isfinite(log_s):
            raise NumericalPrecisionError("Positive success probability underflow")
        result = np.r_[success*probs, -math.expm1(log_s)]
        result.setflags(write=False)
        return result

    def distance(self,p,q,panel,group_features=False):
        if not panel:
            raise ValueError("A shared nonempty probe panel is required")
        groups = self._feature_groups if group_features else None
        distances = []
        for b,c in panel:
            delta = self.full_distribution(tuple(p),b,c)-self.full_distribution(tuple(q),b,c)
            if groups is not None:
                delta = np.array([delta[idx].sum() for idx in groups]+[delta[-1]])
            distances.append(float(np.abs(delta).sum()/2))
        return math.fsum(distances)/len(distances)

def read_word_vector_reference(kernel,p,b,g,rng):
    """Pre-optimization sampler, retained for exact stream-by-stream regression."""
    log_c, cumulative = kernel._word_table(tuple(p),b,tuple(g))
    if log_c == -math.inf:
        return None
    if log_c < 0 and math.log(max(float(rng.random()),np.nextafter(0.,1.))) >= log_c:
        return None
    indices = (rng.random(len(g))[:,None] >= cumulative).sum(axis=1)
    return tuple(kernel.residues[min(int(i),len(kernel.residues)-1)] for i in indices)

def read_word_reference(kernel,p,b,g,rng):
    """Independent position-by-position reference for the same event law."""
    output = []
    for c in g:
        probabilities = kernel.full_distribution(tuple(p),b,c)
        selected = int(rng.choice(len(probabilities),p=probabilities))
        if selected == len(kernel.residues):
            return None
        output.append(kernel.residues[selected])
    return tuple(output)
