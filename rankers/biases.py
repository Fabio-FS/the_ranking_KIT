from __future__ import annotations
from typing import Callable
import numpy as np
from .agents import Agents
from .config import Config

# signature: (agents, received, llr, cfg) -> (w_compat, w_source, w_salience)
BiasFn = Callable[[Agents, np.ndarray, np.ndarray, Config], np.ndarray]


BIASES: dict[str, BiasFn] = {}


def register_bias(name: str) -> Callable[[BiasFn], BiasFn]:
    def decorator(fn: BiasFn) -> BiasFn:
        BIASES[name] = fn
        return fn
    return decorator


def compose_biases(names):
    fns = [BIASES[name] for name in names]

    def combined(agents, received, llr, cfg):
        w = 1.0
        for fn in fns:
            w = w * fn(agents, received, llr, cfg)
        return w

    return combined


@register_bias("baseline")
def bias_baseline(agents, received, llr, cfg):
    return 1.0


@register_bias("confirmation")
def bias_confirmation(agents, received, llr, cfg):
    agreement = np.tanh(agents.beliefs) * np.tanh(llr[received])
    return np.exp(cfg.confirmation_strength * agreement)


@register_bias("negativity")
def bias_negativity(agents, received, llr, cfg):
    return np.where(llr[received] < 0, cfg.negativity_multiplier, 1.0)


@register_bias("illusory_truth")
def bias_illusory_truth(agents, received, llr, cfg):
    count = agents.seen[np.arange(cfg.n), received].astype(np.float64)
    r = 1.0 - 1.0 / cfg.illusory_truth_factor
    return r ** count


@register_bias("conservatism")
def bias_conservatism(agents, received, llr, cfg):
    return 1.0 / (1.0 + cfg.conservatism_strength * np.abs(agents.beliefs))