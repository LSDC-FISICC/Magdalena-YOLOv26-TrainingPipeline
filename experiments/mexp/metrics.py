"""Binary classification metrics, confidence intervals and paired tests (numpy only)."""

import math

import numpy as np


def confusion(y_true, y_pred, positive="plant"):
    t = np.asarray(y_true) == positive
    p = np.asarray(y_pred) == positive
    tp = int((t & p).sum()); tn = int((~t & ~p).sum())
    fp = int((~t & p).sum()); fn = int((t & ~p).sum())
    return tp, fp, fn, tn


def _safe(a, b):
    return a / b if b else float("nan")


def summary(y_true, y_pred, positive="plant"):
    tp, fp, fn, tn = confusion(y_true, y_pred, positive)
    n = tp + fp + fn + tn
    prec = _safe(tp, tp + fp)
    rec = _safe(tp, tp + fn)          # sensitivity: plant seen -> valve opened
    spec = _safe(tn, tn + fp)         # bare soil seen -> valve closed (the saving)
    f1 = _safe(2 * prec * rec, prec + rec)
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    acc = _safe(tp + tn, n)
    lo, hi = wilson_ci(tp + tn, n)
    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": acc, "acc_ci_low": lo, "acc_ci_high": hi,
        "precision": prec, "recall": rec, "specificity": spec, "f1": f1,
        "balanced_accuracy": (rec + spec) / 2,
        "mcc": _safe(tp * tn - fp * fn, mcc_den),
    }


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def bootstrap_ci(y_true, y_pred, metric="accuracy", n_boot=2000, seed=0, positive="plant"):
    rng = np.random.default_rng(seed)
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yt), len(yt))
        vals.append(summary(yt[idx], yp[idx], positive)[metric])
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def paired_accuracy_diff_ci(y_true, pred_a, pred_b, n_boot=2000, seed=0):
    """Bootstrap 95% CI of acc(a) - acc(b) on the same images (in percentage points)."""
    rng = np.random.default_rng(seed)
    yt, a, b = map(np.asarray, (y_true, pred_a, pred_b))
    ca, cb = (a == yt), (b == yt)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yt), len(yt))
        diffs.append(ca[idx].mean() - cb[idx].mean())
    return 100 * float(np.percentile(diffs, 2.5)), 100 * float(np.percentile(diffs, 97.5))


def mcnemar(y_true, pred_a, pred_b):
    """Exact (binomial) McNemar test on discordant pairs. Returns (b, c, p_value).

    b = images only A gets right, c = images only B gets right.
    """
    yt, a, b_ = map(np.asarray, (y_true, pred_a, pred_b))
    ca, cb = (a == yt), (b_ == yt)
    b = int((ca & ~cb).sum())
    c = int((~ca & cb).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return b, c, min(1.0, 2 * p)


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    labels = sorted(set(a) | set(b))
    po = float((a == b).mean())
    pe = sum(float((a == l).mean()) * float((b == l).mean()) for l in labels)
    return _safe(po - pe, 1 - pe), po
