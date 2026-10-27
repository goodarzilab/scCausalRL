"""Learned goal-conditioned GNN RETRIEVER — the deployable, answer-agnostic version of the (answer-informed)
driver_pool. driver_pool proved a small driver-containing action space unblocks from-scratch RL discrimination
(dp_pow2: Briggs drv +0.217); but it reads the answer key. causal_gate tried to reproduce it answer-agnostically
via A-reachability and FAILED (A is dead: ~0-6% driver retention). This module LEARNS the map goal -> driver
shortlist, supervised on TRAIN-fate drivers, and GENERALIZES to held-out fates (predicting their shortlist from
the goal, never seeing held-out answers). If it generalizes, the RL policy explores within a learned driver-
enriched shortlist -> deployable discrimination.

Difference from the in-policy GNN SAMPLER (grn_gene slot-head): the sampler biases per-TF logits within the FULL
280 (exploration still faces 280-choose-k); the retriever PRE-SELECTS a per-goal ~K shortlist, shrinking the
action space BEFORE the policy explores. Different from driver_pool (uses labels) and causal_gate (dead A).

Architecture: per-candidate node feats = [learned cand-embedding, single-TF->goal SCM-reachability (rel0),
gene-space single-TF-Δ cos-to-goal (cos_c)] fused with a goal embedding (from the latent program weights),
message-passed mp_hops over the candidate GRN (D_c A^T D_c^T), -> per-candidate driver logit. BCE vs driver labels.
"""
import numpy as np
import torch
import torch.nn as nn


class LearnedRetriever(nn.Module):
    def __init__(self, C, d, H=128, hops=2):
        super().__init__()
        self.C, self.d, self.hops = C, d, hops
        self.cand_emb = nn.Embedding(C, H)
        self.goal_mlp = nn.Sequential(nn.Linear(d, H), nn.ReLU(), nn.Linear(H, H))
        self.in_mlp = nn.Sequential(nn.Linear(H + H + 2, H), nn.ReLU(), nn.Linear(H, H))   # [cand_emb, goal, rel0, cos]
        self.out = nn.Linear(H, 1)
        self.skip = nn.Parameter(torch.zeros(1))                                            # learned weight on the rel0 hand-feature

    def forward(self, prog_w, rel0, cos_c, GRN):
        """prog_w [d] goal program weights; rel0 [C], cos_c [C] goal-conditioned candidate feats; GRN [C,C] row-norm.
        Returns logits [C]."""
        C = self.C; ci = torch.arange(C, device=prog_w.device)
        g = self.goal_mlp(prog_w).unsqueeze(0).expand(C, -1)                                 # [C,H]
        x = self.in_mlp(torch.cat([self.cand_emb(ci), g, rel0.unsqueeze(1), cos_c.unsqueeze(1)], -1))  # [C,H]
        acc = x
        for _ in range(self.hops):
            x = GRN @ x; acc = acc + x                                                       # message-pass over candidate GRN
        return self.out(acc).squeeze(-1) + self.skip * rel0                                 # [C]


def _goal_feats(op, data, cand, cent_g, G_infl, prog_M, zsingle_gene, dev):
    """Per-goal candidate features. cent_g [G] fate target; returns prog_w [d], rel0 [C], cos_c [C]."""
    cg = cent_g.to(dev)
    prog_w = (prog_M * cg.abs().unsqueeze(0)).sum(1); prog_w = prog_w / (prog_w.sum() + 1e-9)   # [d]
    return prog_w


def build_retriever_gate(op, data, cand, cent, TR, HO, names, dev, *, epochs=300, keep=0.25, hops=2, H=128, lr=1e-3):
    """Train the retriever on TR-fate drivers, return (gate_dict {li: set(local idx)} for ALL goals, stats).
    Answer-agnostic for HO: the retriever never sees HO driver labels."""
    torch.manual_seed(0)                                                                    # DETERMINISTIC init: train-run and eval-run rebuild the IDENTICAL shortlist (policy trained within it)
    C = len(cand); d = op.d
    A = op.mean_A().detach().to(dev)
    if A.dim() == 3: A = A[0]
    Adir = A.t()
    dm = getattr(op.model.decoder, 'dec_mask', None)
    D = (op.model.decoder.D * dm) if dm is not None else op.model.decoder.D
    D = D.detach().to(dev)                                                                  # [G,d]
    cand_t = torch.tensor([int(c) for c in cand], device=dev, dtype=torch.long)
    Dc = D[cand_t, :]                                                                       # [C,d]
    _rn = lambda M: M / (M.abs().sum(1, keepdim=True) + 1e-8)
    GRN = _rn(Dc @ Adir @ Dc.t() + torch.eye(C, device=dev))                                # [C,C] row-norm
    prog_M = D.t()                                                                          # [d,G] program mask (SENA decoder)
    I = torch.eye(d, device=dev)
    try: G_infl = torch.linalg.inv(I - A.t())
    except Exception: G_infl = torch.linalg.inv(I - A.t() + 1e-3 * I)
    with torch.no_grad():
        shifts = op.shift_for(cand).detach().to(dev)                                        # [C,d] single-TF latent shift
        prop = shifts.abs() @ G_infl.abs().t()                                              # [C,d] reachability
        gene_single = shifts @ D.t()                                                        # [C,G] predicted single-TF gene Δ
        gene_single_n = gene_single / (gene_single.norm(dim=1, keepdim=True) + 1e-9)

    # precompute per-goal (prog_w, rel0, cos_c) for ALL goals
    feats = {}
    for li in range(len(names)):
        cg = cent[li].to(dev)
        prog_w = (prog_M * cg.abs().unsqueeze(0)).sum(1); prog_w = prog_w / (prog_w.sum() + 1e-9)   # [d]
        rel0 = (prop * prog_w.unsqueeze(0)).sum(1)                                           # [C]
        rel0 = rel0 / (rel0.max() + 1e-9)
        cgn = cg / (cg.norm() + 1e-9)
        cos_c = (gene_single_n @ cgn)                                                        # [C]
        feats[li] = (prog_w.detach(), rel0.detach(), cos_c.detach())

    # driver labels (local idx) per goal — TRAIN ONLY used for supervision
    n2l = {data.gene_names[int(cand[i])]: i for i in range(C)}
    def _lab(li):
        drv = data.lineage_drivers[names[li]]
        s = set(drv.get('train', [])) | set(drv.get('test', [])) | set(drv.get('full', []))
        return torch.tensor([1.0 if data.gene_names[int(cand[i])] in s else 0.0 for i in range(C)], device=dev)
    Y = {li: _lab(li) for li in TR}
    npos = float(sum(y.sum() for y in Y.values())); nall = float(len(TR) * C)
    pos_w = torch.tensor([max(1.0, (nall - npos) / (npos + 1e-9))], device=dev)             # class imbalance
    import copy
    net = LearnedRetriever(C, d, H=H, hops=hops).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    rng = np.random.default_rng(0); TRl = list(TR)
    net.train(); best_bce = float('inf'); best_state = copy.deepcopy(net.state_dict())    # KEEP-BEST: return the lowest-BCE epoch (BCE spikes on CPU => don't ship the last, ship the best)
    for ep in range(epochs):
        rng.shuffle(TRl); tot = 0.0
        for li in TRl:
            prog_w, rel0, cos_c = feats[li]
            logit = net(prog_w, rel0, cos_c, GRN)
            loss = lossfn(logit, Y[li])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)                          # STABILITY: clip (BCE was exploding to 7e4 => reproducibility gap)
            opt.step(); tot += float(loss)
        tot /= len(TRl)
        if tot < best_bce: best_bce = tot; best_state = copy.deepcopy(net.state_dict())
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"[retriever] ep {ep:4d} BCE {tot:.4f} (best {best_bce:.4f})", flush=True)
    net.load_state_dict(best_state)                                                         # ship the best-fit model, not the last

    # predict gate for ALL goals; report HO generalization (driver-recall of the shortlist, no labels used for HO)
    net.eval(); gate = {}; k = max(1, int(keep * C))
    ho_rec, ho_sz, tr_rec = [], [], []
    with torch.no_grad():
        for li in range(len(names)):
            prog_w, rel0, cos_c = feats[li]
            logit = net(prog_w, rel0, cos_c, GRN)
            top = set(torch.argsort(-logit)[:k].tolist())
            gate[li] = top
            drv = data.lineage_drivers[names[li]]
            dl = {n2l[g] for g in (set(drv.get('train', [])) | set(drv.get('test', [])) | set(drv.get('full', []))) if g in n2l}
            if dl:
                rec = len(dl & top) / len(dl)
                if li in HO: ho_rec.append(rec); ho_sz.append(len(top))
                if li in TR: tr_rec.append(rec)
    stats = dict(k=k, C=C, ho_driver_recall=float(np.mean(ho_rec)) if ho_rec else float('nan'),
                 tr_driver_recall=float(np.mean(tr_rec)) if tr_rec else float('nan'), n_ho=len(ho_rec))
    print(f"[retriever] GENERALIZATION | shortlist k={k}/{C} | HO driver-recall (answer-agnostic) = {stats['ho_driver_recall']:.3f} "
          f"(n={stats['n_ho']}) | TR driver-recall (fit) = {stats['tr_driver_recall']:.3f}", flush=True)
    return gate, stats
