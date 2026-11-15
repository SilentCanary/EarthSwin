"""Pixel and object metrics (docs/07). Streaming histograms => exact PR curve over any number of pixels."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


class PixelMetrics:
    def __init__(self, bins: int = 2000):
        self.bins = bins
        self.pos = np.zeros(bins, np.int64); self.neg = np.zeros(bins, np.int64)

    def update(self, p: np.ndarray, label: np.ndarray):
        """p [...] in [0,1]; label {0,1,255}."""
        v = label != 255
        idx = np.minimum((p[v] * self.bins).astype(np.int64), self.bins - 1)
        yy = label[v] == 1
        self.pos += np.bincount(idx[yy], minlength=self.bins)
        self.neg += np.bincount(idx[~yy], minlength=self.bins)

    def curves(self):
        tp = np.cumsum(self.pos[::-1]); fp = np.cumsum(self.neg[::-1])        # threshold from high to low
        P, N = self.pos.sum(), self.neg.sum()
        fn = P - tp
        prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0)
        rec = tp / max(P, 1)
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        iou = tp / np.maximum(tp + fp + fn, 1)
        thr = 1.0 - (np.arange(self.bins) + 1) / self.bins                      # lower edge of the bin
        return dict(tp=tp, fp=fp, fn=fn, prec=prec, rec=rec, f1=f1, iou=iou, thr=np.maximum(thr, 0.0), P=P, N=N)

    def compute(self, thr: float | None = None) -> dict:
        c = self.curves()
        if c["P"] == 0:
            return dict(ap=float("nan"), f1=float("nan"), iou=float("nan"), thr=float("nan"), n_pos=0)
        rec_prev = np.concatenate([[0.0], c["rec"][:-1]])
        ap = float(((c["rec"] - rec_prev) * c["prec"]).sum())
        k = int(np.argmax(c["f1"]))
        out = dict(ap=ap, f1=float(c["f1"][k]), iou=float(c["iou"][k]), thr=float(c["thr"][k]),
                   prec=float(c["prec"][k]), rec=float(c["rec"][k]), n_pos=int(c["P"]), n_neg=int(c["N"]))
        t = 0.5 if thr is None else thr
        j = min(int(round((1 - t) * self.bins)) - 1, self.bins - 1)
        j = max(j, 0)
        out.update(f1_at=float(c["f1"][j]), iou_at=float(c["iou"][j]), thr_at=t)
        out["ece"] = self.ece()
        return out

    def ece(self, nb: int = 15) -> float:
        """Expected calibration error with equal-width bins on p (positives vs total in each bin)."""
        edges = np.linspace(0, self.bins, nb + 1).astype(int)
        tot = (self.pos + self.neg).sum()
        e = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            n = (self.pos[a:b] + self.neg[a:b]).sum()
            if n == 0:
                continue
            mean_p = ((np.arange(a, b) + 0.5) / self.bins * (self.pos[a:b] + self.neg[a:b])).sum() / n
            e += abs(self.pos[a:b].sum() / n - mean_p) * n / tot
        return float(e)


class ObjectMetrics:
    """Connected-component detection F1 per chip: a predicted object is correct if IoU >= iou_thr with a GT object."""

    def __init__(self, iou_thr: float = 0.1, min_px: int = 9):
        self.iou_thr, self.min_px = iou_thr, min_px
        self.tp = self.fp = self.fn = 0

    def update(self, pred: np.ndarray, label: np.ndarray):
        """pred [B,H,W] bool; label [B,H,W] {0,1,255}."""
        for pr, lb in zip(pred, label):
            gl, ng = ndi.label(lb == 1); pl, npred = ndi.label(pr & (lb != 255))
            gsz = np.bincount(gl.ravel(), minlength=ng + 1); psz = np.bincount(pl.ravel(), minlength=npred + 1)
            matched_g = set(); tp = 0
            for i in range(1, npred + 1):
                if psz[i] < self.min_px:
                    continue
                m = pl == i
                ids, cnt = np.unique(gl[m], return_counts=True)
                ok = False
                for g, inter in zip(ids, cnt):
                    if g == 0:
                        continue
                    if inter / (psz[i] + gsz[g] - inter) >= self.iou_thr:
                        ok = True; matched_g.add(int(g))
                tp += ok
                self.fp += (not ok)
            self.tp += tp
            self.fn += max(ng - len(matched_g), 0)

    def compute(self):
        p = self.tp / max(self.tp + self.fp, 1); r = self.tp / max(self.tp + self.fn, 1)
        return dict(obj_prec=p, obj_rec=r, obj_f1=2 * p * r / max(p + r, 1e-9))
