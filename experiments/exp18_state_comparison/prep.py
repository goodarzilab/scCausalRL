"""Prep: write STATE's 2000-HVG axis + the fair held-out pert sets (perts held out by BOTH
STATE and each world model) for the exp13/exp16-vs-STATE K562 comparison."""
import os, sys, json, re, pickle
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, 'results')
os.makedirs(R, exist_ok=True)
SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--arcinstitute--ST-SE-Replogle/snapshots/e324967ff4cea5ec199e29bcbb5c1f00e5b9d69c")

# STATE 2000 HVG
vd = pickle.load(open(f"{SNAP}/fewshot/k562/var_dims.pkl", "rb"))
state_hvg = list(map(str, vd['gene_names']))
json.dump(state_hvg, open(f"{R}/state_hvg.json", "w"))

# STATE k562 held-out test perts (canonical Replogle K562 test set)
toml = open("/home/sgoudarzi/state-mcp-local/state_preprint_toml_files/replogle_tomls/k562.toml").read()
state_test = set(re.findall(r"'([^']+)'", re.search(r"test\s*=\s*\[(.*?)\]", toml, re.S).group(1))) - {'non-targeting'}

# exp13 test split (reconstruct seed-42 over K562 essential perts)
sys.path.insert(0, '/home/sgoudarzi/scCausalRL/experiments/exp13_crosscell_rpe1')
from data.load_norman import split_single_perts
import anndata as ad
a = ad.read_h5ad("/large_storage/ctc/userspace/aadduri/revisions/replogle_unfiltered/K562_essential_raw_singlecell_01.h5ad", backed='r')
vc = a.obs['gene'].astype(str).value_counts()
single = {g: n for g, n in vc.items() if g != 'non-targeting' and n >= 5}
e13_test = set(split_single_perts(single, 0.8, 0.1, seed=42)['test'])

# exp16 test split (saved in checkpoint)
e16 = torch.load('/home/sgoudarzi/scCausalRL/experiments/exp16_multicellline/results/grid/exp16_multicellline/checkpoints/phase1_best.pt', map_location='cpu', weights_only=False)
e16_test = set(map(str, e16['pert_split']['test']))

exp13_fair = sorted(state_test & e13_test)
exp16_fair = sorted(state_test & e16_test)
json.dump(exp13_fair, open(f"{R}/perts_exp13.json", "w"))
json.dump(exp16_fair, open(f"{R}/perts_exp16.json", "w"))
json.dump(sorted(state_test), open(f"{R}/perts_state_all.json", "w"))
print(f"STATE HVG: {len(state_hvg)} | STATE test: {len(state_test)}")
print(f"exp13 fair (STATE∩exp13): {len(exp13_fair)} | exp16 fair (STATE∩exp16): {len(exp16_fair)}")
