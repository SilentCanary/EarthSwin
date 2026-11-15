"""Block 14: multi-objective loss  L = l1*cls + l2*physics + l3*contrastive + l4*uncertainty (docs/05 §14)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import LossCfg
from .data.formats import IDX_SLOPE
from .models.fusion import alpha_entropy


def focal_loss(p, y, valid, gamma, alpha):
    p = p.clamp(1e-6, 1 - 1e-6)
    pt = p * y + (1 - p) * (1 - y)
    at = alpha * y + (1 - alpha) * (1 - y)
    fl = -at * (1 - pt) ** gamma * pt.log()
    return (fl * valid).sum() / valid.sum().clamp_min(1.0)


def dice_loss(p, y, valid, eps=1.0):
    p, y = p * valid, y * valid
    return 1 - (2 * (p * y).sum() + eps) / (p.sum() + y.sum() + eps)


def edl_loss(alpha, beta, y, valid, epoch, cfg: LossCfg):
    """Beta evidential loss (Bayes risk) + annealed KL to Beta(1,1); positives up-weighted (rare class)."""
    S = alpha + beta
    nll = y * (torch.digamma(S) - torch.digamma(alpha)) + (1 - y) * (torch.digamma(S) - torch.digamma(beta))
    a_t = y + (1 - y) * alpha                      # remove evidence for the correct class
    b_t = (1 - y) + y * beta
    St = a_t + b_t
    kl = (torch.lgamma(St) - torch.lgamma(a_t) - torch.lgamma(b_t)
          + (a_t - 1) * (torch.digamma(a_t) - torch.digamma(St))
          + (b_t - 1) * (torch.digamma(b_t) - torch.digamma(St)))
    lam = min(1.0, epoch / max(cfg.edl_anneal_epochs, 1))
    w = (1 + (cfg.edl_pos_w - 1) * y) * valid
    return ((nll + lam * kl) * w).sum() / w.sum().clamp_min(1.0)


def supcon(emb, label, tau=0.1, n_each=64):
    """Supervised contrastive on sampled stride-4 pixel embeddings: landslide vs background, different locations."""
    B, D, h, w = emb.shape
    y = (label == 1).float().unsqueeze(1)
    v = (label != 255).float().unsqueeze(1)
    pos = F.adaptive_avg_pool2d(y, (h, w))[:, 0] > 0.5
    neg = (F.adaptive_avg_pool2d(y, (h, w))[:, 0] == 0) & (F.adaptive_avg_pool2d(v, (h, w))[:, 0] > 0.99)
    z = emb.permute(0, 2, 3, 1).reshape(-1, D)
    pi, ni = pos.reshape(-1).nonzero().squeeze(1), neg.reshape(-1).nonzero().squeeze(1)
    if len(pi) < 2 or len(ni) < 2:
        return emb.sum() * 0
    pi = pi[torch.randperm(len(pi), device=pi.device)[:n_each]]
    ni = ni[torch.randperm(len(ni), device=ni.device)[:n_each]]
    idx = torch.cat([pi, ni]); lab = torch.cat([torch.ones(len(pi)), torch.zeros(len(ni))]).to(emb.device)
    z = F.normalize(z[idx].float(), dim=1)
    sim = z @ z.T / tau
    eye = torch.eye(len(idx), device=emb.device, dtype=torch.bool)
    logp = sim.masked_fill(eye, float("-inf")); logp = logp - logp.logsumexp(1, keepdim=True)
    same = (lab[:, None] == lab[None]) & ~eye
    return -((logp.masked_fill(~same, 0.0)).sum(1) / same.sum(1).clamp_min(1)).mean()


class PMGSTLoss:
    def __init__(self, cfg: LossCfg):
        self.cfg = cfg

    def __call__(self, out: dict, b: dict, epoch: int = 0):
        c = self.cfg
        lab = b["label"]
        valid = (lab != 255).float()
        y = (lab == 1).float()
        p = out["p"].float()
        parts = {}
        parts["focal"] = focal_loss(p, y, valid, c.focal_gamma, c.focal_alpha)
        parts["dice"] = dice_loss(p, y, valid)
        cls = parts["focal"] + parts["dice"]
        total = c.lam_cls * cls
        # physics: implausibility + factor-of-safety consistency
        phys = p.new_zeros(())
        if out.get("fs_wet") is not None and c.lam_phys > 0:
            slope = b["dem"][:, IDX_SLOPE]
            implaus = ((slope < c.beta_min_deg) | (b["water"][:, 0] > 0.5)).float()
            imp = (p * implaus * valid).sum() / valid.sum().clamp_min(1.0)
            pp = torch.sigmoid(c.fs_kappa * (1 - out["fs_wet"][:, 0].float())).clamp(1e-5, 1 - 1e-5)
            bce = -(y * pp.log() + (1 - y) * (1 - pp).log())
            fsl = 0.5 * ((bce * y * valid).sum() / (y * valid).sum().clamp_min(1.0)
                         + (bce * (1 - y) * valid).sum() / ((1 - y) * valid).sum().clamp_min(1.0))
            parts["implaus"], parts["fs"] = imp, fsl
            phys = imp + 0.1 * fsl
        total = total + c.lam_phys * phys
        con = supcon(out["emb"].float(), lab) if c.lam_con > 0 else p.new_zeros(())
        parts["supcon"] = con
        total = total + c.lam_con * con
        if "alpha" in out and c.lam_unc > 0:
            unc = edl_loss(out["alpha"].float(), out["beta"].float(), y, valid, epoch, c)
            parts["edl"] = unc
            total = total + c.lam_unc * unc
        if c.lam_ent > 0 and out.get("alphas"):
            ent = torch.stack([alpha_entropy(a, pr) for a, pr in out["alphas"]]).mean()
            parts["alpha_ent"] = ent
            total = total - c.lam_ent * ent
        parts["total"] = total
        return total, parts
