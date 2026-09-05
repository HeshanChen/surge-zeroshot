"""All-window vs continuous-window evaluation, side by side, for every headline checkpoint (audit-2, 2026-09-05).
Reads outputs/eval_full_<tag>.log and outputs/eval_full_<tag>_cont.log. -> outputs/compare_cont.txt"""
import re, numpy as np
ROOT = '/Users/heshan/Desktop/surge_fm'
TAGS = [('lstmq_v2final', 'deployment 84'), ('lstmq_v2rotjp', 'rotation Japan'), ('lstmq_v2roteu', 'rotation Europe'),
        ('lstmq_v2rotna', 'rotation N America'), ('lstmq_v2rotocr2', 'rotation Oceania (median run)'), ('lstmq_v2eu299', 'Europe 299 arm'),
        ('lstmq_v2g64', 'ladder 64'), ('lstmq_v2g128', 'ladder 128'), ('lstmq_v2g256', 'ladder 256'), ('lstmq_v2g384', 'ladder 384'),
        ('lstmq_v2g512', 'ladder 512'), ('lstmq_v2g640', 'ladder 640'), ('lstmq_v2c298strat', '298 stratified'), ('lstmq_v2c298usjp', '298 USA+JPN')]
def parse(path):
    try: t = open(path).read()
    except FileNotFoundError: return None
    g = lambda pat, cast=float: (lambda m: cast(m.group(1)) if m else np.nan)(re.search(pat, t))
    return dict(n=g(r'n=(\d+) gauges', int), nwin=g(r'\[W\] Window continuity: (\d+) windows', int), cont=g(r'(\d+) fully continuous', int),
                pooled=g(r'pooled\s+[\d.]+\s+([+-]?\d+)%'), sk8=g(r'L=8\s+[\d.]+\s+([+-]?\d+)%'), nnse8=g(r'L=8\s+[\d.]+\s+[+-]?\d+%\s+([\d.]+)'),
                win999=g(r'p99\.9\s+\d+\s+[\d.]+\s+[\d.]+\s+([+-]?\d+)%'), hw999=g(r'timestep-level.*?\n(?:.*\n){2}\s*p99\.9\s+\d+\s+[\d.]+\s+[\d.]+\s+([+-]?\d+)%'),
                pkcap=g(r'model\s+: bias .*?capture ([\d.]+)'), q99cap=g(r'q99 peak-hour envelope capture .*?: ([\d.]+)'),
                pkcov=g(r'EVENT-LEVEL peak coverage: .*?in ([\d.]+)% of'), tail99=g(r'TAIL coverage .*?: q99 ([\d.]+)'))
lines = ['ALL WINDOWS  vs  CONTINUOUS WINDOWS ONLY   (pooled% | @8h% | NNSE@8h | p99.9 window% | high-water% | peak capture | q99 capture | peak COVERED% | tail cov)', '']
for tag, lab in TAGS:
    a, c = parse(f'{ROOT}/outputs/eval_full_{tag}.log'), parse(f'{ROOT}/outputs/eval_full_{tag}_cont.log')
    if a is None or c is None: lines.append(f'{lab:20s} (missing)'); continue
    f = lambda d: f"{d['pooled']:+4.0f} | {d['sk8']:+4.0f} | {d['nnse8']:.3f} | {d['win999']:+4.0f} | {d['hw999']:+4.0f} | {d['pkcap']:.2f} | {d['q99cap']:.2f} | {d['pkcov']:5.1f} | {d['tail99']:.2f}"
    lines.append(f"{lab:20s} all  {f(a)}   (windows {a['nwin']}, {100*a['cont']/max(a['nwin'],1):.0f}% continuous)")
    lines.append(f"{'':20s} cont {f(c)}")
    lines.append('')
txt = '\n'.join(lines); open(f'{ROOT}/outputs/compare_cont.txt', 'w').write(txt+'\n'); print(txt)
