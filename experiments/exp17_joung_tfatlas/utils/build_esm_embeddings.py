"""Precompute ESM2 protein embeddings for the Joung gene axis → data/joung_esm2_emb.npz
{genes, emb}. Used by the 'esm' shift-encoder descriptor (a sequence→function TF-identity
signal that generalises to held-out TFs). One-time GPU job.

  gene symbol → Swiss-Prot canonical sequence (human_sprot.fasta GN= field)
  sequence → ESM2-t33 (650M, cached) mean-pooled last-hidden → 1280-d
Genes without a sequence get a zero row (descriptor builder handles it).
"""
import os, sys, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
FASTA = os.path.join(ROOT, 'data', 'human_sprot.fasta')
OUT = os.path.join(ROOT, 'data', 'joung_esm2_emb.npz')
MODEL = 'facebook/esm2_t33_650M_UR50D'


def parse_fasta(path):
    seqs = {}
    gene, buf = None, []
    for line in open(path):
        if line.startswith('>'):
            if gene and buf:
                s = ''.join(buf)
                if gene not in seqs or len(s) > len(seqs[gene]):   # keep longest isoform
                    seqs[gene] = s
            buf = []
            gene = None
            for tok in line.split():
                if tok.startswith('GN='):
                    gene = tok[3:].upper()
        else:
            buf.append(line.strip())
    if gene and buf:
        s = ''.join(buf); seqs[gene] = s if (gene not in seqs or len(s) > len(seqs[gene])) else seqs[gene]
    return seqs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(ROOT, 'results/grid/exp17_joung_tfatlas/checkpoints/phase1_epoch80.pt'))
    ap.add_argument('--max_len', type=int, default=1022)
    ap.add_argument('--batch', type=int, default=8)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    gene_names = list(map(str, torch.load(args.ckpt, map_location='cpu', weights_only=False)['gene_names']))
    seqs = parse_fasta(FASTA)
    have = [g for g in gene_names if g.upper() in seqs]
    print(f"genes={len(gene_names)} | with Swiss-Prot sequence: {len(have)}", flush=True)

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModel.from_pretrained(MODEL).to(dev).eval()
    dim = mdl.config.hidden_size
    emb = {}
    with torch.no_grad():
        for i in range(0, len(have), args.batch):
            chunk = have[i:i + args.batch]
            batch_seqs = [seqs[g.upper()][:args.max_len] for g in chunk]
            enc = tok(batch_seqs, return_tensors='pt', padding=True, truncation=True, max_length=args.max_len).to(dev)
            out = mdl(**enc).last_hidden_state                      # [b, L, dim]
            mask = enc['attention_mask'].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)  # mean over residues
            for g, v in zip(chunk, pooled.cpu().numpy()):
                emb[g] = v.astype(np.float32)
            if i % 200 == 0: print(f"  {i}/{len(have)}", flush=True)
    genes = list(emb.keys())
    M = np.stack([emb[g] for g in genes]).astype(np.float32)
    np.savez(OUT, genes=np.array(genes), emb=M)
    print(f"saved {OUT}: {M.shape} ({len(genes)} genes, dim {dim})")


if __name__ == '__main__':
    main()
