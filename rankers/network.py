from __future__ import annotations
import numpy as np
import igraph as ig
from .config import Config


def _build_csr(g: ig.Graph) -> tuple[np.ndarray, np.ndarray]:
    """
    Pack igraph neighbour lists into a CSR layout.
    neighbors_flat[offsets[i] : offsets[i+1]]  gives all neighbours of node i.
    """
    adj = g.get_adjlist()
    degrees = np.array([len(x) for x in adj], dtype=np.int32)
    offsets = np.zeros(len(adj) + 1, dtype=np.int32)
    np.cumsum(degrees, out=offsets[1:])
    if offsets[-1] > 0:
        flat = np.concatenate([np.asarray(x, dtype=np.int32) for x in adj])
    else:
        flat = np.empty(0, dtype=np.int32)
    return flat, offsets


def build_network(cfg: Config) -> tuple[ig.Graph, np.ndarray, np.ndarray]:
    """Return (graph, neighbors_flat, neighbor_offsets) for a WS instance."""
    g = ig.Graph.Watts_Strogatz(dim=1, size=cfg.n, nei=cfg.k // 2, p=cfg.p_rewire)
    flat, offsets = _build_csr(g)
    return g, flat, offsets


def build_neighbor_table(flat: np.ndarray, offsets: np.ndarray, n: int) -> np.ndarray:
    """
    Pad ragged CSR neighbour lists into a dense (N, max_degree) table.
    Empty pad slots point to row index n (the dummy zero-weight row in Agents).
    """
    deg = offsets[1:] - offsets[:-1]
    max_deg = int(deg.max())
    nb_pad = np.full((n, max_deg), n, dtype=np.int32)
    for i in range(n):
        lo, hi = offsets[i], offsets[i + 1]
        nb_pad[i, :hi - lo] = flat[lo:hi]
    return nb_pad