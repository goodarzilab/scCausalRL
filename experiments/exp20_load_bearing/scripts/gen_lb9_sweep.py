"""Generate Sweep-1 (MMD x DEG-MSE) and Sweep-3 (nlSCM + MMD) configs from the proven bases.
Sweep-1 base = masked_tgt_delta_plus_mmd (masked_go + mmd=1 + structural + gate, prior-off, no retrieval).
Sweep-3 base = nlSCM_refresh_k20 (nonlinear SCM); we set MMD on and DROP retrieval.
"""
import os, copy, yaml
BASE = "configs/ablations"
OUT = os.path.join(BASE, "lb9sweep"); os.makedirs(OUT, exist_ok=True)
s1 = yaml.safe_load(open(f"{BASE}/lb7v2b/v2b_masked_tgt_delta_plus_mmd.yaml"))
s3 = yaml.safe_load(open(f"{BASE}/lb8dfix/v2b_nlSCM_refresh_k20.yaml"))


def write(base, name, p1over):
    c = copy.deepcopy(base)
    c['phase1'].update(p1over)
    c['phase1']['lambda_prior'] = 0.0                    # prior OFF (canonical)
    c['wandb']['phase1_run_name'] = f"exp20_{name}"
    yaml.safe_dump(c, open(f"{OUT}/{name}.yaml", 'w'), sort_keys=False)
    print(name, "->", {k: c['phase1'].get(k) for k in
          ('pert_mmd_weight', 'pert_deg_mse_weight', 'retrieval', 'scm_nonlinear')})


# ---- Sweep 1: MMD optimum (deg=0; mmd=1 already exists) + DEG-MSE stacking ----
write(s1, "v2b_mmd0p5", dict(pert_mmd_weight=0.5))
write(s1, "v2b_mmd2",   dict(pert_mmd_weight=2.0))
write(s1, "v2b_mmd5",   dict(pert_mmd_weight=5.0))
write(s1, "v2b_mmd1_deg2", dict(pert_mmd_weight=1.0, pert_deg_mse_weight=2.0, pert_deg_mse_thr=0.1))
write(s1, "v2b_mmd2_deg2", dict(pert_mmd_weight=2.0, pert_deg_mse_weight=2.0, pert_deg_mse_thr=0.1))

# ---- Sweep 3: nlSCM + MMD, retrieval DROPPED ----
noretr = dict(retrieval=False)
for k in ('retrieval_k', 'retrieval_tau', 'retrieval_key_dim', 'retrieval_build_epoch', 'retrieval_refresh_every'):
    s3['phase1'].pop(k, None)
write(s3, "v2b_nlSCM_mmd1", dict(pert_mmd_weight=1.0, **noretr))
write(s3, "v2b_nlSCM_mmd2_deg2", dict(pert_mmd_weight=2.0, pert_deg_mse_weight=2.0, pert_deg_mse_thr=0.1, **noretr))
print("\nwrote configs to", OUT)
