"""Aggregate the 3-seed frame-routed random-null eval (job 2753443 log) into mean±sd per metric,
split by source (ALL/joung/briggs) + HSPC. Feeds the FDR H-step deck's frame-routed results slides.
Usage: python agg_frameroute_null.py <log.out>"""
import re, sys, json
import numpy as np

log = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/sgoudarzi/scCausalRL/experiments_rl/logs/rl_frameroute_null_2753443.out"
txt = open(log).read()

# split into per-seed blocks on the "##### FRAME-ROUTED SEED n #####" markers the launcher echoes
blocks = re.split(r'#####\s*FRAME-ROUTED SEED (\d+)\s*#####', txt)
# blocks = [preamble, '0', body0, '1', body1, '2', body2]
seeds = {}
for i in range(1, len(blocks) - 1, 2):
    seeds[int(blocks[i])] = blocks[i + 1]

def grab_source(body, src):
    m = re.search(rf'^\s*{src}\s+n=\d+ \| drv_recall20=\S+\(Δ([+\-][\d.]+)\)\s+soft1=\S+\(Δ([+\-][\d.]+)\)\s+soft2=\S+\(Δ([+\-][\d.]+)\)', body, re.M)
    if not m: return None
    return dict(drv_recall20=float(m[1]), soft1=float(m[2]), soft2=float(m[3]))

def grab_scalar(body, pat):
    m = re.search(pat, body); return float(m[1]) if m else None

rows = {}
for s, body in sorted(seeds.items()):
    r = {}
    # held-out headline
    m = re.search(r'held-out:policy \(amortized\)\] N=256.*?specificity \+?([\d.\-]+).*?drv_recall@20 ([\d.]+)', body, re.S)
    if m: r['heldout_spec'] = float(m[1]); r['heldout_drv20'] = float(m[2])
    for src in ('ALL', 'joung', 'briggs'):
        g = grab_source(body, src)
        if g:
            for k, v in g.items(): r[f'{src}_{k}'] = v
    r['hspc_hard_d'] = grab_scalar(body, r'HSPC HARD\].*?policy ([\d.]+) vs null [\d.]+±[\d.]+ \(Δ([+\-][\d.]+)\)')
    m = re.search(r'HSPC HARD\].*?\(Δ([+\-][\d.]+)\)', body); r['hspc_hard_d'] = float(m[1]) if m else None
    m = re.search(r'HSPC SOFT\].*?SOFT-1 policy \+?[\d.\-]+ vs null \+?[\d.\-]+ \(Δ([+\-][\d.]+)\)', body); r['hspc_soft1_d'] = float(m[1]) if m else None
    m = re.search(r'HSPC SOFT\].*?SOFT-2 policy \+?[\d.\-]+ vs null \+?[\d.\-]+ \(Δ([+\-][\d.]+)\)', body); r['hspc_soft2_d'] = float(m[1]) if m else None
    m = re.search(r'ARRAYED-SOFT\].*?SOFT-1 policy \+?[\d.\-]+ vs null \+?[\d.\-]+ \(Δ([+\-][\d.]+)\)', body); r['arr_soft1_d'] = float(m[1]) if m else None
    m = re.search(r'ARRAYED-SOFT\].*?SOFT-2 policy \+?[\d.\-]+ vs null \+?[\d.\-]+ \(Δ([+\-][\d.]+)\)', body); r['arr_soft2_d'] = float(m[1]) if m else None
    m = re.search(r'ARRAYED-Jaccard\].*?policy ([\d.]+) vs null [\d.]+±[\d.]+ \(Δ([+\-][\d.]+)\)', body); r['arr_jac_d'] = float(m[2]) if m else None
    rows[s] = r

keys = sorted({k for r in rows.values() for k in r})
print(f"[agg] parsed {len(rows)} seeds: {sorted(rows)}\n")
agg = {}
print(f"{'metric':<20} " + "  ".join(f"s{s}" for s in sorted(rows)) + "   mean±sd")
for k in keys:
    vals = [rows[s].get(k) for s in sorted(rows) if rows[s].get(k) is not None]
    if not vals: continue
    mu, sd = float(np.mean(vals)), float(np.std(vals))
    agg[k] = dict(mean=round(mu, 3), sd=round(sd, 3), n=len(vals), vals=[round(v, 3) for v in vals])
    print(f"{k:<20} " + "  ".join(f"{rows[s].get(k, float('nan')):+.3f}" for s in sorted(rows)) + f"   {mu:+.3f}±{sd:.3f}")

out = log.replace('.out', '_agg.json')
json.dump({'per_seed': rows, 'agg': agg}, open(out, 'w'), indent=2)
print(f"\n[agg] wrote {out}")
