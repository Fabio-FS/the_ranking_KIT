from __future__ import annotations
import numpy as np
import numba
import math


@numba.njit(parallel=True, cache=True)
def _fused_baseline_col(nb_table, own_msgid, read_ring, gumbel, out_col):
    """
    For each agent, pick one unread neighbour-message column via Gumbel-top-1
    over uniform weights. Fuses gather + already-read masking + top-k draw into
    a single parallel pass with no (N, C, 2W) temporaries.

    out_col[i] = winning slot index (nbi * W + col), matching the column space
    of the old gather_candidates / draw_without_replacement output.
    """
    n, d = nb_table.shape
    w = own_msgid.shape[1]
    r = read_ring.shape[1]
    for i in numba.prange(n):
        best_key = -1e30
        best_col = 0
        for nbi in range(d):
            sender = nb_table[i, nbi]
            for col in range(w):
                msgid = own_msgid[sender, col]
                if msgid < 0:                      # pad slot
                    continue
                seen = False
                for j in range(r):                 # already-read check
                    if read_ring[i, j] == msgid:
                        seen = True
                        break
                if seen:
                    continue
                slot = nbi * w + col
                key = gumbel[i, slot]
                if key > best_key:
                    best_key = key
                    best_col = slot
        out_col[i] = best_col


@numba.njit(parallel=True, cache=True)
def _fused_similarity_col(nb_table, own_msgid, own_claim, read_ring,
                          beliefs, llr, gumbel, out_col):
    """
    Weighted Gumbel-top-1 with weight = exp(-|belief_i - LLR(claim)|).
    Key = log(weight) + gumbel = -|belief_i - LLR(claim)| + gumbel.
    """
    n, d = nb_table.shape
    w = own_msgid.shape[1]
    r = read_ring.shape[1]
    for i in numba.prange(n):
        b = beliefs[i]
        best_key = -1e30
        best_col = 0
        for nbi in range(d):
            sender = nb_table[i, nbi]
            for col in range(w):
                msgid = own_msgid[sender, col]
                if msgid < 0:
                    continue
                seen = False
                for j in range(r):
                    if read_ring[i, j] == msgid:
                        seen = True
                        break
                if seen:
                    continue
                claim = own_claim[sender, col]
                log_weight = -abs(b - llr[claim])
                slot = nbi * w + col
                key = log_weight + gumbel[i, slot]
                if key > best_key:
                    best_key = key
                    best_col = slot
        out_col[i] = best_col


@numba.njit(parallel=True, cache=True)
def _fused_engagement_col(nb_table, own_msgid, read_ring,
                          liked_count, gumbel, out_col):
    """
    Weighted Gumbel-top-1 with weight = 1 + liked_count[i, sender].
    Key = log(weight) + gumbel.
    """
    n, d = nb_table.shape
    w = own_msgid.shape[1]
    r = read_ring.shape[1]
    for i in numba.prange(n):
        best_key = -1e30
        best_col = 0
        for nbi in range(d):
            sender = nb_table[i, nbi]
            affinity = liked_count[i, sender]
            log_weight = math.log(1.0 + affinity)
            for col in range(w):
                msgid = own_msgid[sender, col]
                if msgid < 0:
                    continue
                seen = False
                for j in range(r):
                    if read_ring[i, j] == msgid:
                        seen = True
                        break
                if seen:
                    continue
                slot = nbi * w + col
                key = log_weight + gumbel[i, slot]
                if key > best_key:
                    best_key = key
                    best_col = slot
        out_col[i] = best_col


@numba.njit(parallel=True, cache=True)
def _fused_post_popularity_col(nb_table, own_msgid, own_likes, read_ring,
                               gumbel, out_col):
    """
    Weighted Gumbel-top-1 with weight = 1 + likes on the message.
    Key = log(weight) + gumbel.
    """
    n, d = nb_table.shape
    w = own_msgid.shape[1]
    r = read_ring.shape[1]
    for i in numba.prange(n):
        best_key = -1e30
        best_col = 0
        for nbi in range(d):
            sender = nb_table[i, nbi]
            for col in range(w):
                msgid = own_msgid[sender, col]
                if msgid < 0:
                    continue
                seen = False
                for j in range(r):
                    if read_ring[i, j] == msgid:
                        seen = True
                        break
                if seen:
                    continue
                log_weight = math.log(1.0 + own_likes[sender, col])
                slot = nbi * w + col
                key = log_weight + gumbel[i, slot]
                if key > best_key:
                    best_key = key
                    best_col = slot
        out_col[i] = best_col


@numba.njit(parallel=True, cache=True)
def _fused_user_popularity_col(nb_table, own_msgid, read_ring,
                               user_likes, gumbel, out_col):
    """
    Weighted Gumbel-top-1 with weight = 1 + lifetime likes of the sender.
    Key = log(weight) + gumbel.
    """
    n, d = nb_table.shape
    w = own_msgid.shape[1]
    r = read_ring.shape[1]
    for i in numba.prange(n):
        best_key = -1e30
        best_col = 0
        for nbi in range(d):
            sender = nb_table[i, nbi]
            log_weight = math.log(1.0 + user_likes[sender])
            for col in range(w):
                msgid = own_msgid[sender, col]
                if msgid < 0:
                    continue
                seen = False
                for j in range(r):
                    if read_ring[i, j] == msgid:
                        seen = True
                        break
                if seen:
                    continue
                slot = nbi * w + col
                key = log_weight + gumbel[i, slot]
                if key > best_key:
                    best_key = key
                    best_col = slot
        out_col[i] = best_col