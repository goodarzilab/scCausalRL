"""M2 — Set-based autoregressive policy + critic.

The policy emits an UNORDERED SET of ≤ k_max perturbations applied simultaneously
(matching how a Norman combo is delivered). It is drawn as an autoregressive token
recipe over the vocabulary V = {STOP} ∪ candidate_genes (single-direction / CRISPRa,
so |V| = C+1; the KO/OE doubling of the general spec collapses to the live half):

  - shared encoder  h = enc([z_t ; z_target])  feeds all k_max slots (BDQ-style),
    so cost grows LINEARLY in k_max, not combinatorially;
  - π_θ(a|s) = ∏_i π_θ(a_i | s, a_{<i}); each factor is a masked softmax over V from
    [h ; Σ_{j<i} e(g_j) ; slot_emb(i)] — the chosen-gene context is an ORDER-INVARIANT
    sum of gene embeddings (matches set semantics);
  - masks: STOP forbidden at slot 0 (no empty action); already-chosen genes forbidden
    at later slots (no duplicates);
  - log π = Σ_i log-softmax(a_i);  entropy = Σ_i per-slot entropy.

`sample` (collection) and `evaluate` (PPO update, batched) return the summed log-prob
and summed entropy — the only quantities PPO consumes, so the PPO core is unchanged.
Token 0 = STOP; token t≥1 = candidate gene index t-1.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e9


class SetPolicy(nn.Module):
    def __init__(self, state_dim, n_candidates, k_max=2, emb_dim=64, hidden=256, slot_dim=16,
                 slot_head='mlp', mp_hops=0, mp_gate='gated', mp_op=None, decode_W=None):
        super().__init__()
        self.C = int(n_candidates)
        self.k_max = int(k_max)
        self.V = self.C + 1                       # +1 for STOP (index 0)
        self.enc = nn.Sequential(
            nn.Linear(2 * state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.gene_emb = nn.Embedding(self.C, emb_dim)       # e(g) for the context sum
        self.slot_emb = nn.Embedding(self.k_max, slot_dim)
        feat = hidden + emb_dim + slot_dim
        self.head = nn.Sequential(
            nn.Linear(feat, hidden), nn.ReLU(),
            nn.Linear(hidden, self.V))
        self.emb_dim = emb_dim
        # ---- GRAPH message-passing slot head (optional): propagate per-slot scores k hops over the WM's learned
        # causal graph (fixed A^T operator), then read out candidate logits. Injects the WM mechanism as an
        # action-space inductive bias. mlp=current; grn_latent=MP in 357-factor(GO) space then SENA-decode;
        # grn_gene=MP directly on the candidate-gene GRN. gate: 'gated' = MLP base + beta*MP (residual); 'none' = pure MP. ----
        self.slot_head = str(slot_head); self.mp_hops = int(mp_hops); self.mp_gate = str(mp_gate)
        if self.slot_head != 'mlp':
            assert mp_op is not None, "graph slot head needs mp_op"
            self.register_buffer('mp_op', mp_op.float())              # [N,N] fixed row-normalized propagation operator
            self.stop_head = nn.Linear(feat, 1)                      # STOP logit (graph head doesn't produce it)
            if self.slot_head == 'grn_latent':
                assert decode_W is not None, "grn_latent needs decode_W (SENA decoder rows for candidates)"
                self.register_buffer('decode_W', decode_W.float())   # [C, d]
                self.factor_head = nn.Linear(feat, mp_op.shape[0])   # feat -> d factor(GO-program) scores
            elif self.slot_head == 'grn_gene':
                self.cand_head = nn.Linear(feat, self.C)             # feat -> C candidate scores (MP on [C,C] GRN)
            else:
                raise ValueError(f"unknown slot_head {slot_head}")
            if self.mp_gate == 'gated':
                self.mp_beta = nn.Parameter(torch.tensor(0.5))       # learnable residual gate on the MP path

    def encode(self, s_emb, t_emb):
        return self.enc(torch.cat([s_emb, t_emb], -1))      # [B, hidden]

    def _slot_logits(self, h, ctx_sum, slot_i, chosen_mask):
        """h [B,H], ctx_sum [B,emb], slot_i int, chosen_mask [B,C] bool (True=taken).
        Returns masked logits [B, V]."""
        B = h.shape[0]
        slot = self.slot_emb(torch.full((B,), slot_i, dtype=torch.long, device=h.device))
        feat = torch.cat([h, ctx_sum, slot], -1)
        if self.slot_head == 'mlp':
            logits = self.head(feat)                                 # [B, V]
        else:
            if self.slot_head == 'grn_latent':
                s = self.factor_head(feat); acc = s                  # [B, d]
                for _ in range(self.mp_hops):
                    s = s @ self.mp_op.t(); acc = acc + s            # k-hop propagate over A^T
                cand = acc @ self.decode_W.t()                       # [B, C]  SENA-decode to candidate genes
            else:                                                    # grn_gene: MP directly on the candidate GRN
                s = self.cand_head(feat); acc = s                    # [B, C]
                for _ in range(self.mp_hops):
                    s = s @ self.mp_op.t(); acc = acc + s
                cand = acc                                           # [B, C]
            mp_logits = torch.cat([self.stop_head(feat), cand], -1)  # [B, V]
            logits = (self.head(feat) + self.mp_beta * mp_logits) if self.mp_gate == 'gated' else mp_logits
        if slot_i == 0:
            logits[:, 0] = NEG_INF                               # no STOP at slot 0
        if chosen_mask is not None and chosen_mask.any():
            logits[:, 1:] = logits[:, 1:].masked_fill(chosen_mask, NEG_INF)
        ab = getattr(self, '_allowed_block', None)                   # per-goal RETRIEVER/gate shortlist: [C] bool (True=blocked); applies in train (sample) + eval (select) via this one choke point
        if ab is not None:
            logits[:, 1:] = logits[:, 1:].masked_fill(ab.unsqueeze(0), NEG_INF)
        return logits

    @torch.no_grad()
    def sample(self, s_emb, t_emb):
        """Single state (B=1). Returns (tokens list incl. trailing STOP if any,
        gene_set list[int local], logp_sum float, entropy_sum float)."""
        h = self.encode(s_emb.unsqueeze(0), t_emb.unsqueeze(0))   # [1,H]
        ctx = torch.zeros(1, self.emb_dim, device=h.device)
        chosen_mask = torch.zeros(1, self.C, dtype=torch.bool, device=h.device)
        tokens, genes = [], []
        logp_sum, ent_sum = 0.0, 0.0
        for i in range(self.k_max):
            logits = self._slot_logits(h, ctx, i, chosen_mask)    # [1,V]
            dist = torch.distributions.Categorical(logits=logits)
            tok = int(dist.sample()[0])
            logp_sum += float(dist.log_prob(torch.tensor([tok], device=h.device))[0])
            ent_sum += float(dist.entropy()[0])
            tokens.append(tok)
            if tok == 0:                                          # STOP
                break
            g = tok - 1
            genes.append(g)
            chosen_mask[0, g] = True
            ctx = ctx + self.gene_emb(torch.tensor([g], device=h.device))
        # pad tokens to k_max with STOP(0) for storage
        tokens = tokens + [0] * (self.k_max - len(tokens))
        return tokens[:self.k_max], genes, logp_sum, ent_sum

    def evaluate(self, s_emb, t_emb, token_rows):
        """Batched re-evaluation for PPO. token_rows [B, k_max] long (0=STOP pad).
        Returns (logp_sum [B], entropy_sum [B]) — sums over slots up to & incl. the
        first STOP (STOP's own log-prob is included; padding after STOP is ignored)."""
        B = s_emb.shape[0]
        h = self.encode(s_emb, t_emb)                             # [B,H]
        ctx = torch.zeros(B, self.emb_dim, device=h.device)
        chosen_mask = torch.zeros(B, self.C, dtype=torch.bool, device=h.device)
        logp = torch.zeros(B, device=h.device)
        ent = torch.zeros(B, device=h.device)
        active = torch.ones(B, dtype=torch.bool, device=h.device)  # not yet stopped
        for i in range(self.k_max):
            logits = self._slot_logits(h, ctx, i, chosen_mask)    # [B,V]
            dist = torch.distributions.Categorical(logits=logits)
            tok = token_rows[:, i]                                # [B]
            lp = dist.log_prob(tok)
            e = dist.entropy()
            logp = logp + torch.where(active, lp, torch.zeros_like(lp))
            ent = ent + torch.where(active, e, torch.zeros_like(e))
            is_stop = (tok == 0)
            # update context for non-stop, active rows
            took_gene = active & (~is_stop)
            if took_gene.any():
                g = (tok - 1).clamp(min=0)
                add = self.gene_emb(g) * took_gene.unsqueeze(-1).float()
                ctx = ctx + add
                chosen_mask = chosen_mask | (F.one_hot(g, self.C).bool() & took_gene.unsqueeze(-1))
            active = active & (~is_stop)
        return logp, ent


class SetCritic(nn.Module):
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, s_emb, t_emb):
        return self.net(torch.cat([s_emb, t_emb], -1)).squeeze(-1)
