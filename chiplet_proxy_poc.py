#!/usr/bin/env python3
"""Two-partition exact attention proof-of-concept.

This script demonstrates a minimal "multi-die proxy" experiment:
- Baseline exact SDPA computed monolithically.
- Two-partition exact SDPA where K/V are statically sharded and each partition
  returns only online-softmax reduction state.

The final output should match baseline up to floating-point tolerance.
"""

import argparse
import math
import random
from typing import List, Tuple


Matrix = List[List[float]]


def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def random_matrix(rows: int, cols: int, scale: float = 1.0) -> Matrix:
    return [[(2.0 * random.random() - 1.0) * scale for _ in range(cols)] for _ in range(rows)]


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def sdpa_baseline(q: Matrix, k: Matrix, v: Matrix) -> Matrix:
    """Exact SDPA with stable softmax, output shape [S, Dv]."""
    seq_len = len(q)
    out_dim = len(v[0])
    out = zeros(seq_len, out_dim)

    for i in range(seq_len):
        logits = [dot(q[i], k[j]) for j in range(seq_len)]
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        denom = sum(exps)
        for j in range(seq_len):
            w = exps[j] / denom
            for d in range(out_dim):
                out[i][d] += w * v[j][d]
    return out


def local_partition_state(
    q_row: List[float], k_part: Matrix, v_part: Matrix
) -> Tuple[float, float, List[float]]:
    """Return local online-softmax state (m, l, o) for one shard."""
    logits = [dot(q_row, k_vec) for k_vec in k_part]
    m = max(logits)
    exp_terms = [math.exp(x - m) for x in logits]
    l = sum(exp_terms)
    out_dim = len(v_part[0])
    o = [0.0 for _ in range(out_dim)]
    for idx, e in enumerate(exp_terms):
        for d in range(out_dim):
            o[d] += e * v_part[idx][d]
    return m, l, o


def sdpa_two_partition_proxy(q: Matrix, k: Matrix, v: Matrix) -> Tuple[Matrix, int]:
    """Exact SDPA by reducing two shard-local online-softmax states."""
    seq_len = len(q)
    out_dim = len(v[0])
    out = zeros(seq_len, out_dim)

    split = seq_len // 2
    k0, k1 = k[:split], k[split:]
    v0, v1 = v[:split], v[split:]

    # Track "cross-partition words" exchanged:
    # per query, each partition shares (m, l, o[dv]) => (2 + Dv) scalars.
    words_exchanged = 0

    for i in range(seq_len):
        m0, l0, o0 = local_partition_state(q[i], k0, v0)
        m1, l1, o1 = local_partition_state(q[i], k1, v1)
        words_exchanged += 2 * (2 + out_dim)

        m = max(m0, m1)
        a0 = math.exp(m0 - m)
        a1 = math.exp(m1 - m)
        l = a0 * l0 + a1 * l1

        for d in range(out_dim):
            out[i][d] = (a0 * o0[d] + a1 * o1[d]) / l

    return out, words_exchanged


def max_abs_diff(a: Matrix, b: Matrix) -> float:
    diff = 0.0
    for r in range(len(a)):
        for c in range(len(a[0])):
            diff = max(diff, abs(a[r][c] - b[r][c]))
    return diff


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-partition attention chiplet proxy PoC")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--d-model", type=int, default=32, help="Q/K dimension")
    parser.add_argument("--d-value", type=int, default=32, help="V/output dimension")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    if args.seq_len < 2 or args.seq_len % 2 != 0:
        raise ValueError("--seq-len must be an even integer >= 2")

    random.seed(args.seed)
    q = random_matrix(args.seq_len, args.d_model, scale=0.2)
    k = random_matrix(args.seq_len, args.d_model, scale=0.2)
    v = random_matrix(args.seq_len, args.d_value, scale=0.2)

    baseline = sdpa_baseline(q, k, v)
    proxy, words_state = sdpa_two_partition_proxy(q, k, v)
    err = max_abs_diff(baseline, proxy)

    # Naive communication baselines:
    # 1) score-only exchange: one scalar per key score per query.
    # 2) score+value-vector exchange: one score + Dv-vector per key per query.
    words_logits = args.seq_len * args.seq_len
    words_score_value = args.seq_len * args.seq_len * (1 + args.d_value)
    reduction_logits = words_logits / max(1, words_state)
    reduction_score_value = words_score_value / max(1, words_state)

    print("=== Chiplet Proxy Attention PoC ===")
    print(f"shape: S={args.seq_len}, D={args.d_model}, Dv={args.d_value}")
    print(f"max |baseline - proxy|: {err:.3e}")
    print(f"state words exchanged: {words_state}")
    print(f"score-only words exchanged (naive): {words_logits}")
    print(f"score+value words exchanged (naive): {words_score_value}")
    print(f"score-only/state exchange ratio: {reduction_logits:.2f}x")
    print(f"score+value/state exchange ratio: {reduction_score_value:.2f}x")
    print("PASS" if err < 1e-6 else "WARN: tolerance exceeded")


if __name__ == "__main__":
    main()
