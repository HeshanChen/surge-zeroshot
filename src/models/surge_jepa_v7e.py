"""v7e: the hourly-forcing transformer with forcing tokens that carry hour and channel identity.

Audit 2026-09-04 (docs/protocol_symmetry_audit.md, section 4): SurgeJEPA_v7 tokenises the 4 x 48 forcing
values with one Linear(1, D) per scalar and concatenates them into the key/value set with no hour embedding
and no channel embedding, so its output is invariant to reversing the forcing hours or permuting channels
(max |dy| ~ 1e-6). It therefore cannot integrate forcing in time, which is the very capability the factor
study attributes to recurrence. v7e adds a learned hour embedding (48) and a learned channel embedding (4)
to each forcing token; everything else (encoder, predictor, quantile heads, zero-initialised residual head,
loss) is inherited unchanged from v7. Same CFG, same param count up to the two small embedding tables."""
import os as _os
_ROOT = _os.environ.get('SURGE_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
import torch, torch.nn as nn
from models.surge_jepa_v7 import SurgeJEPA_v7, loss_v7, TAUS

class SurgeJEPA_v7e(SurgeJEPA_v7):
    def __init__(s, **kw):
        super().__init__(**kw)
        s.f_h_emb = nn.Embedding(48, s.D)                     # which forecast hour a forcing token belongs to
        s.f_ch_emb = nn.Embedding(4, s.D)                     # which forcing variable it is (wu, wv, mslp, precip)
        nn.init.normal_(s.f_h_emb.weight, std=0.02); nn.init.normal_(s.f_ch_emb.weight, std=0.02)
    def predict_window(s, ctx, ff, Tt, static, anchor=None):
        z, ch, T = s.encode_context(ctx, static); cs = z[:, (ch == 0), :]
        B, Cf, Tf = ff.shape                                   # (B,4,48)
        dev = ff.device
        ftok = s.ftok1(ff.reshape(B, Cf, Tf, 1))               # (B,4,48,D)
        ftok = ftok + s.f_ch_emb(torch.arange(Cf, device=dev)).view(1, Cf, 1, s.D) \
                    + s.f_h_emb(torch.arange(Tf, device=dev)).view(1, 1, Tf, s.D)
        ftok = ftok.reshape(B, Cf*Tf, s.D)
        kv = s.pin(torch.cat([cs, ftok], 1))
        q = s.qry[:, :Tf, :].expand(B, Tf, -1) + s.h_emb(torch.arange(1, Tf+1, device=dev).float().view(1, Tf, 1))
        gb = s.pfilm(static)
        for i, (at, fn, n1, n2) in enumerate(zip(s.pred, s.pff, s.pn1, s.pn2)):
            q = q + at(n1(q), kv, kv)[0]; q = q + fn(n2(q)); q = q*(1+gb[:, i, 0:1, :]) + gb[:, i, 1:2, :]
        q = s.pfn(q); zh = s.pout(q)
        delta = s.ground(zh)
        return zh, (delta if anchor is None else anchor[:, None, None] + delta)

def loss_v7e(model, ctx, ff, tgt, static, anchor, lam_q=1.0):
    return loss_v7(model, ctx, ff, tgt, static, anchor, lam_q=lam_q)
