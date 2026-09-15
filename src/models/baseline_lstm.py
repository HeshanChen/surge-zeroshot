"""Global LSTM baseline — THE falsifier for the SSL/transformer story (docs/next_steps_sota.md #1).

Nagaraj&Wahl-2025-style regional recipe scaled global: seq2seq LSTM, static attrs concatenated to every
input step (their conditioning), trained on the SAME 298-station split / weighted loss / residual
parameterization as Surge-JEPA v1. If this matches v1's zero-shot, the SSL story dies — we must know.

Deliberately STRONG, not a strawman: hidden 512 x 2 layers (~6.5M), encoder state initializes decoder,
zero-init output head (starts AT persistence, same discipline as v1), same magnitude-weighted MSE.
What it lacks vs v1 (the actual hypothesis under test): JEPA SSL objective, transformer attention, FiLM.

API mirrors SurgeJEPA_v1.predict_window so scripts/eval_full.py drives it unchanged (--model lstm):
  predict_window(ctx (B,5,Tctx), ff (B,4,48), Tt, static (B,5), anchor (B)) -> (None, (B,48))
  .L = 1 so eval_full's H//model.L == 48.
"""
import os as _os
_ROOT = _os.environ.get('SURGE_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
import torch, torch.nn as nn

class GlobalLSTM(nn.Module):
    def __init__(s, C=5, n_static=5, hidden=512, layers=2, n_out=1):
        super().__init__(); s.L = 1; s.hidden, s.layers, s.n_out = hidden, layers, n_out
        s.enc = nn.LSTM(C + n_static, hidden, layers, batch_first=True)
        s.dec = nn.LSTM((C - 1) + n_static, hidden, layers, batch_first=True)
        s.head = nn.Linear(hidden, n_out)
        nn.init.zeros_(s.head.weight); nn.init.zeros_(s.head.bias)     # residual: start at persistence
    def predict_window(s, ctx, ff, Tt, static, anchor=None):
        B = ctx.shape[0]
        xe = torch.cat([ctx.transpose(1, 2), static[:, None, :].expand(-1, ctx.shape[2], -1)], -1)   # (B,Tctx,C+ns)
        _, hc = s.enc(xe)
        xd = torch.cat([ff.transpose(1, 2), static[:, None, :].expand(-1, ff.shape[2], -1)], -1)     # (B,48,4+ns)
        hd, _ = s.dec(xd, hc)
        delta = s.head(hd)                                              # (B,48,n_out)
        if s.n_out == 1:
            delta = delta.squeeze(-1)                                   # (B,48)
            return None, (delta if anchor is None else anchor[:, None] + delta)
        return None, (delta if anchor is None else anchor[:, None, None] + delta)   # (B,48,3): point,q90,q99

def lstm_loss(model, ctx, ff, tgt, static, anchor):
    _, ph = model.predict_window(ctx, ff, tgt.shape[1], static, anchor)
    wt = 1.0 + tgt.abs()                                                # same magnitude weight as v1
    return (wt*(ph-tgt)**2).mean()/wt.mean()
