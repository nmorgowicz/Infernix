"""Analysis core of the KV-format evaluation kit (the instrument definition; see docs/maintainer/gemma4-benchmarks.md).

Pure functions over `infernix-perplexity` report.json documents and `--save-top-tokens` files
(native NINFTOPK version 1, k = 64). Nothing here reads a model, tokenizer or corpus.

Definitions (all teacher-forced, one reference arm, identical corpus/context/stride/prefix ids):
  PPL / dPPL%       exp(sum total_nll / sum scored_tokens) per scope; dPPL% = 100 * (PPL_arm / PPL_ref - 1).
  top-1 agreement   fraction of scored positions whose dump rank-0 id equals the reference's rank-0 id.
  truncated KL      KL(P_ref || Q_arm) over the reference's 64-id set S, both renormalised over S; an id of S that is
                    absent from the arm's dump takes the arm's 64th log-probability (upper bound on its true value);
                    probability mass outside S is ignored. A truncated-distribution diagnostic, NOT full-vocabulary KL.
  target dlogprob   arm_logprob(target) - ref_logprob(target), only where the target id is inside BOTH dumps.
  paired t          per-window differences d_w; mean(d) / (std(d, ddof=1) / sqrt(n_windows)).
"""

from __future__ import annotations

import math
import struct
import numpy as np

K = 64
MAGIC = b"NINFTOPK"
REC = np.dtype([("id", "=i4", (K,)), ("lp", "=f4", (K,))])
CHUNK = 20000


class EvalError(ValueError):
    """An input violates a comparability or format requirement; the message names the next action."""


def token_fingerprint(tokens) -> int:
    """Match the native FNV-1a-64 over four little-endian bytes per input token."""
    value = 14695981039346656037
    for token in tokens:
        for byte in int(token).to_bytes(4, "little", signed=False):
            value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
    return value


def open_dump(path, expected_positions: int | None = None, label: str = "", report: dict | None = None,
              expected_fingerprints: list[int] | None = None):
    """Check native headers/layout; optionally require corpus-derived input hashes in window order."""
    name = label or str(path)
    windows = []
    parts = []
    with open(path, "rb") as handle:
        def take(size):
            data = handle.read(size)
            if len(data) != size:
                raise EvalError(f"{name}: truncated top-token reference {path}")
            return data

        def number():
            return struct.unpack("=I", take(4))[0]

        def text():
            size = number()
            if size > 4096:
                raise EvalError(f"{name}: invalid top-token reference string")
            return take(size).decode("utf-8")

        if take(8) != MAGIC or number() != 1:
            raise EvalError(f"{name}: expected infernix-perplexity --save-top-tokens NINFTOPK version 1")
        k, context, stride = number(), number(), number()
        corpus, mode = text(), text()
        if k != K:
            raise EvalError(f"{name}: expected top-{K}, got top-{k}")
        if report is not None:
            execution = report["execution"]
            if (corpus, mode, context, stride) != (
                report["corpus"]["id"], report["corpus"]["mode"],
                execution["context_tokens"], execution["stride_tokens"],
            ):
                raise EvalError(f"{name}: top-token reference header differs from report")
        while True:
            stream = number()
            if stream == 0xffffffff:
                if handle.read(1):
                    raise EvalError(f"{name}: trailing bytes after top-token reference end marker")
                break
            begin, end, fingerprint = struct.unpack("=QQQ", take(24))
            if expected_fingerprints is not None:
                index = len(windows)
                if index >= len(expected_fingerprints):
                    raise EvalError(f"{name}: no reconstructed input fingerprint for window {index}")
                if fingerprint != expected_fingerprints[index]:
                    raise EvalError(f"{name}: input fingerprint differs from reconstructed corpus tokens at window {index}")
            count = end - begin
            if not 0 < count <= 1 << 24:
                raise EvalError(f"{name}: invalid top-token window range {begin}:{end}")
            remaining = path.stat().st_size - handle.tell()
            if count * (4 + K * 8) > remaining:
                raise EvalError(f"{name}: truncated top-token window")
            take(count * 4)  # Target logprobs are retained by the native file, not needed by this diagnostic.
            part = np.empty(count, dtype=REC)
            part["id"] = np.frombuffer(take(count * K * 4), dtype="=i4").reshape(count, K)
            part["lp"] = np.frombuffer(take(count * K * 4), dtype="=f4").reshape(count, K)
            parts.append(part)
            windows.append((stream, begin, end))
    if expected_fingerprints is not None and len(windows) != len(expected_fingerprints):
        raise EvalError(f"{name}: reference and reconstructed input fingerprint counts differ")
    if report is not None:
        expected = [(index, w["target_begin"], w["target_end"])
                    for index, stream in enumerate(report["streams"]) for w in stream["windows"]]
        if windows != expected:
            raise EvalError(f"{name}: top-token window layout differs from report")
    count = sum(len(part) for part in parts)
    if expected_positions is not None and count != expected_positions:
        raise EvalError(f"{name}: reference holds {count} positions, report scored {expected_positions}")
    if not parts:
        raise EvalError(f"{name}: top-token reference contains no positions")
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


def target_lp(dump, targets):
    """Log-prob of the target token if it is inside the dumped top-64, else NaN."""
    out = np.full(len(dump), np.nan, np.float32)
    for a in range(0, len(dump), CHUNK):
        b = min(a + CHUNK, len(dump))
        ids = dump["id"][a:b]
        hit = ids == targets[a:b, None]
        has = hit.any(axis=1)
        pos = hit.argmax(axis=1)
        lp = np.take_along_axis(dump["lp"][a:b], pos[:, None], axis=1)[:, 0]
        out[a:b] = np.where(has, lp, np.nan)
    return out


def per_position(arm, ref, tlp_arm, tlp_ref):
    """Per-position top-1 agreement, truncated KL, missing-id count, bit-identity and target dlogprob."""
    n = len(ref)
    top1 = np.empty(n, bool)
    kl = np.empty(n, np.float64)
    missing = np.empty(n, np.int16)
    exact = np.empty(n, bool)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        ri, rl = ref["id"][a:b], ref["lp"][a:b].astype(np.float64)
        ai, al = arm["id"][a:b], arm["lp"][a:b].astype(np.float64)
        top1[a:b] = ri[:, 0] == ai[:, 0]
        exact[a:b] = (ri == ai).all(axis=1) & (rl.astype(np.float32) == al.astype(np.float32)).all(axis=1)
        eq = ri[:, :, None] == ai[:, None, :]
        found = eq.any(axis=2)
        idx = eq.argmax(axis=2)
        q = np.where(found, np.take_along_axis(al, idx, axis=1), al[:, -1:])
        missing[a:b] = (~found).sum(axis=1)
        p = np.exp(rl)
        qq = np.exp(q)
        p /= p.sum(axis=1, keepdims=True)
        qq /= qq.sum(axis=1, keepdims=True)
        kl[a:b] = (p * (np.log(p) - np.log(qq))).sum(axis=1)
    both = ~np.isnan(tlp_arm) & ~np.isnan(tlp_ref)
    d = np.where(both, tlp_arm.astype(np.float64) - tlp_ref.astype(np.float64), np.nan)
    return {"top1": top1, "kl": kl, "missing": missing, "exact": exact, "both": both, "d": d}


def agg(pp, mask):
    """Summary of the per-position arrays over the positions selected by boolean `mask`."""
    m = mask
    dd = pp["d"][m & pp["both"]]
    return {
        "positions": int(m.sum()),
        "top1_agree": float(pp["top1"][m].mean()),
        "bit_identical_topk_frac": float(pp["exact"][m].mean()),
        "kl_top64_truncated_mean": float(pp["kl"][m].mean()),
        "kl_top64_truncated_p99": float(np.percentile(pp["kl"][m], 99)),
        "kl_top64_truncated_max": float(pp["kl"][m].max()),
        "ref_top64_ids_missing_from_arm_frac": float((pp["missing"][m] > 0).mean()),
        "target_in_both_top64_frac": float((m & pp["both"]).sum() / m.sum()),
        "target_dlogprob_mean_subset": float(dd.mean()) if dd.size else None,
        "target_abs_dlogprob_mean_subset": float(np.abs(dd).mean()) if dd.size else None,
        "target_abs_dlogprob_p99_subset": float(np.percentile(np.abs(dd), 99)) if dd.size else None,
        "target_abs_dlogprob_max_subset": float(np.abs(dd).max()) if dd.size else None,
    }


def paired_t(values):
    """mean, standard error and t of a vector of paired differences; se/t are None when undefined (n < 2, zero spread)."""
    d = np.asarray(values, dtype=np.float64)
    out = {"n": int(d.size), "mean": float(d.mean()), "se": None, "t": None}
    if d.size > 1:
        out["se"] = float(d.std(ddof=1) / math.sqrt(d.size))
        if d.std() > 0:
            out["t"] = float(d.mean() / (d.std(ddof=1) / math.sqrt(d.size)))
    return out


def windows_of(report):
    """{(stream id, target_begin, target_end): (domain, total_nll, scored_tokens)} for every scored window."""
    out = {}
    for s in report["streams"]:
        for w in s["windows"]:
            out[(s["id"], w["target_begin"], w["target_end"])] = (s["domain"], w["total_nll"], w["scored_tokens"])
    return out


def ppl_delta(rep_a, rep_r):
    """Overall and per-domain PPL of both arms over the windows they share, plus paired window dNLL/token stats."""
    wa, wr = windows_of(rep_a), windows_of(rep_r)
    keys = sorted(set(wa) & set(wr))
    if not keys or any(wa[k][2] != wr[k][2] for k in keys):
        raise EvalError("the two reports share no scored windows, or their scored-token counts differ per window")
    res = {"windows": len(keys), "domains": {}}
    for scope in ["overall"] + sorted({wa[k][0] for k in keys}):
        ks = [k for k in keys if scope == "overall" or wa[k][0] == scope]
        n = sum(wa[k][2] for k in ks)
        nll_a, nll_r = sum(wa[k][1] for k in ks), sum(wr[k][1] for k in ks)
        ppl_a, ppl_r = math.exp(nll_a / n), math.exp(nll_r / n)
        dl = np.array([(wa[k][1] - wr[k][1]) / wa[k][2] for k in ks])
        stat = paired_t(dl)
        row = {"scored_tokens": n, "ppl_arm": ppl_a, "ppl_ref": ppl_r, "dppl_pct": 100.0 * (ppl_a / ppl_r - 1.0),
               "windows": len(ks), "mean_dnll_per_token_window_mean": stat["mean"], "se": stat["se"], "t": stat["t"]}
        if scope == "overall":
            res["overall"] = row
        else:
            res["domains"][scope] = row
    return res


def window_bin_stats(win_idx, bin_idx, values, mask, n_bins):
    """Per (window, bin) mean of `values` (NaN where a window has no position in the bin)."""
    key = win_idx.astype(np.int64) * n_bins + bin_idx
    sel = mask & (bin_idx >= 0)
    cnt = np.bincount(key[sel], minlength=(win_idx.max() + 1) * n_bins)
    tot = np.bincount(key[sel], weights=values[sel], minlength=len(cnt))
    cnt = cnt.reshape(-1, n_bins)
    return np.divide(tot.reshape(-1, n_bins), cnt, out=np.full(cnt.shape, np.nan), where=cnt > 0)


def window_means(win_idx, values):
    """Mean of `values` within each global window index."""
    n_win = int(win_idx.max()) + 1
    return np.bincount(win_idx, weights=values, minlength=n_win) / np.bincount(win_idx, minlength=n_win)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p from discordant counts b (A right, B wrong) and c (A wrong, B right).

    p = min(1, 2 * P(Binomial(b + c, 1/2) <= min(b, c))). With b = c = 0 there is no discordant pair, the test
    carries no information and p = 1.0 by convention (degenerate at a ceiling or floor).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)
