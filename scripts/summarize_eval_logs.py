"""Compact summary of every quantity the manuscript quotes from an eval_full log, for a list of tags, all windows and
continuous windows side by side. Reads outputs/eval_full_<tag>{,_cont}.log.
Usage: python3 scripts/summarize_eval_logs.py [tag ...]   (default: the manuscript's headline tags)
-> prints a table and writes outputs/eval_log_summary.txt"""
import re, sys
ROOT = '/Users/heshan/Desktop/surge_fm'
DEFAULT = ['lstmq_v2final', 'lstmq_v2rotjp', 'lstmq_v2roteu', 'lstmq_v2rotna', 'lstmq_v2rotoc', 'lstmq_v2rotocr2', 'lstmq_v2rotocr3',
           'lstmq_v2eu299', 'lstmq_v2g64', 'lstmq_v2g128', 'lstmq_v2g256', 'lstmq_v2g384', 'lstmq_v2g512', 'lstmq_v2g640',
           'lstmq_v2c298strat', 'lstmq_v2c298usjp', 'lstmq_v2n298', 'v7e_n298v']
tags = sys.argv[1:] or DEFAULT

def grab(txt, pat, cast=float, default=None):
    m = re.search(pat, txt)
    return cast(m.group(1)) if m else default

def parse(path):
    try: t = open(path).read()
    except FileNotFoundError: return None
    r = {}
    r['n'] = grab(t, r'n=(\d+) gauges', int)
    r['cont%'] = grab(t, r'fully continuous \(([\d.]+)%\)')
    r['pooled'] = grab(t, r'pooled\s+[\d.]+\s+(-?\d+)%', int)
    r['nnse_pool'] = grab(t, r'pooled\s+[\d.]+\s+-?\d+%\s+([\d.]+)')
    for L in (4, 8, 12):
        r[f'sk{L}'] = grab(t, rf'L={L}\s+[\d.]+\s+(-?\d+)%', int)
        r[f'nnse{L}'] = grab(t, rf'L={L}\s+[\d.]+\s+-?\d+%\s+([\d.]+)')
    # window-level and high-water skills
    win = re.search(r'window-level.*?\n.*?\n(.*?)timestep-level', t, re.S)
    if win:
        for line in win.group(1).strip().splitlines():
            f = line.split(); r[f'win_{f[0]}'] = int(f[-1].rstrip('%'))
    hw = re.search(r'timestep-level.*?\n(.*?)\n\n', t, re.S)
    if hw:
        for line in hw.group(1).strip().splitlines():
            f = line.split(); r[f'hw_{f[0]}'] = int(f[-1].rstrip('%'))
    r['pk_model'] = grab(t, r'model\s*:.*?capture ([\d.]+)')
    r['pk_pers'] = grab(t, r'persist:.*?capture ([\d.]+)')
    r['n_storm'] = grab(t, r'Peak amplitude .*?n=(\d+)', int)
    for L in (4, 6, 8, 10, 12):
        r[f'st{L}'] = grab(t, rf'L=\s*{L}h: mdl.*?skill\s*([+-]?\d+)%', int)
    r['cov90'] = grab(t, r'q90 ([\d.]+) \(nominal'); r['cov99'] = grab(t, r'q99 ([\d.]+) \(nominal')
    r['tail99'] = grab(t, r'TAIL coverage.*?q99 ([\d.]+)')
    r['sharp'] = grab(t, r'ratio ([\d.]+)')
    r['cross'] = grab(t, r'crossing rate.*?: ([\d.]+)')
    r['q99env'] = grab(t, r'q99 peak-hour envelope capture.*?: ([\d.]+)')
    for b, lab in (('1-6 h', 'b1'), ('7-12h', 'b2'), ('13-24h', 'b3'), ('25-48h', 'b4')):
        r[f'q99env_{lab}'] = grab(t, rf'peak at\s+{re.escape(b)}: q99 capture ([\d.]+)')
        r[f'pk_{lab}'] = grab(t, rf'peak at\s+{re.escape(b)}: n=\s*\d+\s+model capture ([\d.]+)')
        r[f'cov_{lab}'] = grab(t, rf'peak at\s+{re.escape(b)}: covered ([\d.]+)%')
    r['evcov'] = grab(t, r'EVENT-LEVEL peak coverage.*?in ([\d.]+)% of')
    r['evcov10'] = grab(t, r'within 10% below the peak or above: ([\d.]+)%')
    hl = re.search(r'hourly q90 / q99 coverage by lead: (.*)', t)
    if hl: r['hourly'] = hl.group(1).strip()
    return r

KEYS = ['n', 'cont%', 'pooled', 'sk4', 'sk8', 'sk12', 'nnse8', 'win_p99', 'win_p99.9', 'hw_p99', 'hw_p99.5', 'hw_p99.9', 'pk_model', 'pk_pers',
        'st4', 'st6', 'st8', 'st10', 'st12', 'pk_b1', 'pk_b4', 'q99env', 'q99env_b1', 'q99env_b4', 'evcov', 'cov_b1', 'cov_b4',
        'cov90', 'cov99', 'tail99', 'sharp', 'cross']
lines = []
for tag in tags:
    for suf in ('', '_cont'):
        r = parse(f'{ROOT}/outputs/eval_full_{tag}{suf}.log')
        if r is None: continue
        lines.append(f'{tag+suf:28s} ' + ' '.join(f'{k}={r.get(k)}' for k in KEYS))
        if r.get('hourly'): lines.append(f'{"":28s} hourly: {r["hourly"]}')
txt = '\n'.join(lines); print(txt); open(f'{ROOT}/outputs/eval_log_summary.txt', 'w').write(txt + '\n')
