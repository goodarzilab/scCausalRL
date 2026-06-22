"""Deck: Joung TF-atlas world model — annotation (human fetal atlas), singles-only training,
results, epistasis, and next-step architectures (exp17 + exp19).

Reuses the helper style from make_eval_methodology_slides.py (python-pptx + LaTeX→PNG cache).
Annotation numbers are read from analysis/human_dev_annotation/human_annotation_stats.json.
Results/epistasis figures are PLACEHOLDERS with file paths (user inserts the images);
the human-fetal UMAP is embedded if present.

Run: python slides/make_exp17_19_slides.py   →   slides/exp17_19_overview.pptx
"""
import os, json
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'exp17_19_overview.pptx')
EQ_DIR = os.path.join(HERE, '_eq_cache'); os.makedirs(EQ_DIR, exist_ok=True)
FIGJ = os.path.join(ROOT, 'experiments', 'figures', 'joung')
STATS = os.path.join(ROOT, 'experiments', 'exp17_joung_tfatlas', 'analysis',
                     'human_dev_annotation', 'human_annotation_stats.json')

WHITE=RGBColor(0xFF,0xFF,0xFF); BLACK=RGBColor(0,0,0); DARK=RGBColor(0x1A,0x1A,0x2E)
BLUE=RGBColor(0x21,0x71,0xB5); GRAY=RGBColor(0x66,0x66,0x66); LIGHT=RGBColor(0xF5,0xF5,0xF5)
RED=RGBColor(0xC0,0x39,0x2B); GREEN=RGBColor(0x27,0xAE,0x60); AMBER=RGBColor(0xD9,0x82,0x1E)
DARKBLUE=RGBColor(0x0B,0x3D,0x91); PURPLE=RGBColor(0x6B,0x3F,0xA0)

prs = Presentation(); prs.slide_width=Inches(13.333); prs.slide_height=Inches(7.5)


def slide_blank():
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg=s.background; bg.fill.solid(); bg.fill.fore_color.rgb=WHITE; return s

def text(s,l,t,w,h,txt,*,size=18,bold=False,color=BLACK,align=PP_ALIGN.LEFT,italic=False):
    tb=s.shapes.add_textbox(Inches(l),Inches(t),Inches(w),Inches(h)); tf=tb.text_frame; tf.word_wrap=True
    p=tf.paragraphs[0]; p.text=txt; p.font.size=Pt(size); p.font.bold=bold; p.font.italic=italic
    p.font.color.rgb=color; p.font.name='Calibri'; p.alignment=align; return tb

def bullets(s,l,t,w,h,items,*,size=14,color=BLACK,spacing=5):
    tb=s.shapes.add_textbox(Inches(l),Inches(t),Inches(w),Inches(h)); tf=tb.text_frame; tf.word_wrap=True
    for i,it in enumerate(items):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
        p.text='• '+it; p.font.size=Pt(size); p.font.color.rgb=color; p.font.name='Calibri'; p.space_after=Pt(spacing)
    return tb

def title_bar(s,txt,*,color=DARK):
    sh=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(0),Inches(0),Inches(13.333),Inches(0.85))
    sh.fill.solid(); sh.fill.fore_color.rgb=color; sh.line.fill.background()
    text(s,0.5,0.15,12.3,0.6,txt,size=24,bold=True,color=WHITE)

def label(s,l,t,w,txt,*,color=DARKBLUE,size=17):
    tb=s.shapes.add_textbox(Inches(l),Inches(t),Inches(w),Inches(0.4)); tf=tb.text_frame; tf.word_wrap=True
    p=tf.paragraphs[0]; p.text=txt; p.font.size=Pt(size); p.font.bold=True; p.font.color.rgb=color; p.font.name='Calibri'; return tb

def box(s,l,t,w,h,*,fill=LIGHT,line=None):
    sh=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(l),Inches(t),Inches(w),Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb=fill
    if line is None: sh.line.fill.background()
    else: sh.line.color.rgb=line; sh.line.width=Pt(1.5)
    sh.shadow.inherit=False; return sh

def render_eq_block(name,lines,*,fontsize=18,color='#0B3D91',dpi=300,ls=1.5,pad=0.22):
    path=os.path.join(EQ_DIR,f'{name}.png'); fig=plt.figure(figsize=(0.01,0.01)); fig.patch.set_alpha(0)
    ax=fig.add_axes([0,0,1,1]); ax.set_axis_off(); n=len(lines); ts=[]
    for i,ln in enumerate(lines):
        ts.append(ax.text(0.5,1.0-(i+0.5)/n,ln,fontsize=fontsize,color=color,ha='center',va='center'))
    fig.canvas.draw(); mw=max(t.get_window_extent().width for t in ts)/fig.dpi
    h=n*(fontsize/72.0)*ls; fig.set_size_inches(mw+2*pad,h+2*pad)
    fig.savefig(path,dpi=dpi,transparent=True,bbox_inches='tight',pad_inches=pad); plt.close(fig); return path

def eq_block(s,l,t,name,lines,*,width=None,fontsize=18,color='#0B3D91',ls=1.5):
    p=render_eq_block(name,lines,fontsize=fontsize,color=color,ls=ls)
    if width: s.shapes.add_picture(p,Inches(l),Inches(t),width=Inches(width))
    else: s.shapes.add_picture(p,Inches(l),Inches(t))

def fig_or_placeholder(s,l,t,w,h,path,caption):
    """Embed the image if it exists; otherwise draw a dashed placeholder box with the path."""
    if os.path.exists(path):
        s.shapes.add_picture(path,Inches(l),Inches(t),height=Inches(h))
    else:
        sh=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(l),Inches(t),Inches(w),Inches(h))
        sh.fill.solid(); sh.fill.fore_color.rgb=RGBColor(0xFA,0xFA,0xFA)
        sh.line.color.rgb=GRAY; sh.line.width=Pt(1.25); sh.line.dash_style=2 if False else None
        sh.shadow.inherit=False
        text(s,l+0.2,t+h/2-0.45,w-0.4,0.4,'[ FIGURE PLACEHOLDER ]',size=15,bold=True,color=GRAY,align=PP_ALIGN.CENTER)
    text(s,l,t+h+0.03,w,0.35,caption,size=10,italic=True,color=GRAY,align=PP_ALIGN.CENTER)

def table(s,l,t,w,rows,col_w,*,header=True,fs=11,hfs=11):
    nr=len(rows); nc=len(rows[0])
    gt=s.shapes.add_table(nr,nc,Inches(l),Inches(t),Inches(w),Inches(0.3*nr)).table
    for j,cw in enumerate(col_w): gt.columns[j].width=Inches(cw)
    for i,row in enumerate(rows):
        for j,val in enumerate(row):
            c=gt.cell(i,j); c.text=str(val); p=c.text_frame.paragraphs[0]
            p.font.size=Pt(hfs if (header and i==0) else fs); p.font.name='Calibri'
            p.font.bold=(header and i==0)
            p.font.color.rgb=WHITE if (header and i==0) else BLACK
            c.fill.solid(); c.fill.fore_color.rgb=DARKBLUE if (header and i==0) else (LIGHT if i%2 else WHITE)
            c.margin_top=Pt(2); c.margin_bottom=Pt(2)
    return gt


def load_stats():
    if os.path.exists(STATS):
        return json.load(open(STATS))
    return None  # job not finished yet → slide 2 shows a 'pending' note

# ════════════════════════ SLIDE 1 — Title ════════════════════════
s=slide_blank()
text(s,1,1.7,11.3,1.4,'Causal-VAE World Model for the Joung TF Atlas',size=38,bold=True,color=DARK,align=PP_ALIGN.CENTER)
text(s,1,3.0,11.3,1.3,'Developmental annotation (human fetal atlas) · singles-only training & architectures ·\n'
     'held-out singles/combo results · epistasis · next-step architectures',size=18,color=GRAY,align=PP_ALIGN.CENTER)
text(s,1,5.7,11.3,0.4,'scCausalRL  |  exp17_joung_tfatlas + exp19_nonlinear_compose',size=14,color=GRAY,align=PP_ALIGN.CENTER)

# ════════════════════════ SLIDE 2 — Annotation methods + stats ════════════════════════
st=load_stats()
s=slide_blank(); title_bar(s,'1 · Annotating the Joung TF Atlas with the Human Fetal Cell Atlas')
box(s,0.3,1.0,6.4,5.05,fill=LIGHT,line=BLUE)
label(s,0.5,1.1,6.0,'Approach — shift alignment',color=BLUE)
bullets(s,0.5,1.55,6.0,2.6,[
    'Per TF: induced shift Δ = mean(OE cells) − mean(hESC control) — not the absolute profile (shared pluripotency baseline hides identity).',
    'Reference: Cao 2020 human fetal atlas (GSE156793) — 172 organ-cell-type expression signatures Dc = mean(type c) − mean over types.',
    'Score = alignment(Δ, Dc) after stripping the shared OE nuisance axes (top-4 SVD of the centred TF-shift matrix).',
    'Human reference ⇒ NO mouse→human orthology step: direct symbol match keeps ~all 9,923 Joung HVGs (vs the ~15k mouse-ortholog subset).',
],size=12)
eq_block(s,0.5,4.25,'annot_eq',[
    r'$\Delta_{TF}=\mathrm{mean}(\mathrm{OE})-\mathrm{mean}(\mathrm{ctrl})$',
    r'$\mathrm{score}_c=\cos(\,\tilde\Delta_{TF},\,D_c\,),\;\; \mathrm{spec}=\max_c \mathrm{score}_c-\overline{\mathrm{score}}$',
],width=6.0,fontsize=15)

box(s,6.95,1.0,6.05,5.05,fill=LIGHT,line=GREEN)
label(s,7.15,1.1,5.7,'Method shoot-out (chosen: cosine)',color=GREEN)
if st:
    m=st['methods']; thr_key=[k for k in m['cosine'] if k.startswith('n_conf_gt_') and '0.05' not in k]
    tk=thr_key[0] if thr_key else 'n_conf_gt_0.05'; conf=tk.split('_')[-1]
    rows=[['method',f'#conf(>{conf})','#conf(>0.05)','med spec','agree↔cos']]
    for mm in ['cosine','nnls','ols','ridge']:
        d=m[mm]; rows.append([mm,d.get(tk,'-'),d.get('n_conf_gt_0.05','-'),
                              f"{d['median_spec']:.3f}",f"{d.get('agree_top_with_cosine',1.0):.2f}"])
    table(s,7.15,1.55,5.65,rows,[1.5,1.25,1.25,1.0,1.0],fs=11,hfs=11)
    cs=st['cosine_specificity']; fdr=st.get('fdr',{})
    bullets(s,7.15,3.6,5.7,2.4,[
        f"TFs scored (n≥50): {st['n_TFs']}  |  cell types: {st['n_celltypes']}  |  shared genes: {st['shared_genes']}",
        f"cosine specificity: median {cs['median']:.3f}, q90 {cs['q90']:.3f}, max {cs['max']:.3f}",
        f"FDR-controlled (gene-permutation null, {fdr.get('pool_null_size',0):,} draws + BH): q<0.05 → {fdr.get('n_fdr_0.05','?')} TFs, q<0.10 → {fdr.get('n_fdr_0.10','?')} TFs.",
        f"Heuristic spec>{conf} flags {m['cosine'][tk]} (vs 13 with mouse) — but only {fdr.get('n_fdr_0.05','?')}/{fdr.get('n_fdr_0.10','?')} survive FDR: 172 collinear types ⇒ heavy null tail.",
        f"cond(DᵀD)={st['cond_DtD']:.0e} ⇒ NNLS/OLS/ridge degenerate (0 calls); cosine is the stable method.",
    ],size=11)
else:
    text(s,7.15,1.6,5.7,3,'[annotation job results pending —\nslide auto-fills from human_annotation_stats.json]',size=14,color=GRAY)
box(s,0.3,6.25,12.7,0.95,fill=RGBColor(0xEC,0xF7,0xEC),line=GREEN)
text(s,0.55,6.4,12.3,0.6,'Chosen: cosine + permutation-FDR gate. Human ref (no orthology loss) gives far more candidate calls than mouse, '
     'but proper FDR shows the labels are still sparse → use the q<0.10 / top-decile set as the confident pool, with care.',size=12.5,color=DARK)

# ════════════════════════ SLIDE 3 — Human atlas figure ════════════════════════
s=slide_blank(); title_bar(s,'1 · The Human Fetal Reference (Cao 2020 / Descartes, GSE156793)')
fig_or_placeholder(s,0.4,1.1,7.4,5.5,os.path.join(FIGJ,'human_fetal_atlas_umap.png'),
                   'experiments/figures/joung/human_fetal_atlas_umap.png')
box(s,8.1,1.1,4.9,5.4,fill=LIGHT,line=BLUE)
label(s,8.3,1.2,4.5,'Why this reference',color=BLUE)
bullets(s,8.3,1.65,4.5,4.6,[
    '~4M cells, 121 fetal tissues, 15 organs (sci-RNA-seq3).',
    '172 organ-cell-type signatures = the targets Δ_TF is aligned to.',
    'Human → direct symbol match to the Joung 9,923-HVG axis (no mouse orthology loss).',
    'Same reference family the Joung 2023 paper used to map TF profiles to fates → our annotation is comparable to the publication.',
    'Caveat: midgestation fetal is slightly LATER than the early-progenitor states 7-day hESC OE reaches; trajectory tiers differ from MOCA.',
],size=12)

# ════════════════════════ SLIDE 4 — Training data / vocab ════════════════════════
s=slide_blank(); title_bar(s,'2 · Training: Singles-Only on the Joung TF Atlas')
box(s,0.3,1.0,6.4,5.6,fill=LIGHT,line=BLUE)
label(s,0.5,1.1,6.0,'Data & vocabulary',color=BLUE)
bullets(s,0.5,1.55,6.0,3.0,[
    'Raw atlas dense 1.15M × 37,528 (≈172 GB) → chunk-preprocessed to a compact 9,923-gene axis.',
    'Vocabulary (9,923 genes) = 8,506 combinatorial-screen genes ∪ 1,769 TF genes (the on-axis perturbable set).',
    'Full training set: 1,144,579 cells × 9,923 genes; controls = GFP/mCherry cells (~127,813).',
    'Trains on SINGLE TF over-expressions only; combos/triples fully held out for compositional eval.',
],size=12)
label(s,0.5,4.75,6.0,'Splits (gene-level, seed 42)',color=DARKBLUE,size=15)
table(s,0.5,5.15,6.0,[['split','# TFs','use'],
    ['train','1,351','fit the world model'],
    ['val','169','early-stop / selection'],
    ['test','169','held-out single-TF zero-shot'],
    ['combos','44 pairs + 3 triples','held-out composition']],[1.4,1.5,3.0],fs=11)

box(s,6.95,1.0,6.05,5.6,fill=LIGHT,line=PURPLE)
label(s,7.15,1.1,5.7,'The world model (shift-encoder + SCM do-operator)',color=PURPLE,size=15)
bullets(s,7.15,1.55,5.7,1.5,[
    'MLP encoder → linear-Gaussian SCM on latents → LINEAR decoder (kept linear for the GRN + CollecTRI prior).',
    'A held-out TF is predicted zero-shot from its feature (control-PCA or ESM embedding) via the learned shift encoder.',
],size=12)
eq_block(s,7.1,3.15,'wm_eq',[
    r'$z=(I-A^{\top})^{-1}\varepsilon \quad\text{(latent SCM)}$',
    r'$S=\sum_i \mathrm{ShiftEnc}(f_{g_i}),\;\; z_S=(I-A^{\top})^{-1}S$',
    r'$\tilde\varepsilon=u\odot z_{\mathrm{target}}+(1-u)\odot\varepsilon$',
    r'$z_{\mathrm{post}}=(I-\mathrm{diag}(1-u)\,A^{\top})^{-1}\tilde\varepsilon \;\;\text{(do-op)}$',
    r'$\widehat{\Delta x}=D z_{\mathrm{post}}-D z_{\mathrm{prev}}$',
],width=5.9,fontsize=15,ls=1.5)

# ════════════════════════ SLIDE 5 — Architectures + losses ════════════════════════
s=slide_blank(); title_bar(s,'2 · Architectures Tried & Loss Functions')
label(s,0.4,0.95,7,'Ablation grid (shift input × residual target × contrastive × latent)',color=DARK,size=15)
table(s,0.35,1.35,7.4,[
    ['variant','input','resid-MSE','contrastive','note'],
    ['orig','ctrl-PCA','✗','✗','baseline'],
    ['esm','ESM-2','✗','✗','input swap'],
    ['esm_resid','ESM-2','✓','✗','collapsed'],
    ['esm_resid_con','ESM-2','✓','✓','collapsed'],
    ['esm_con  ★','ESM-2','✗','✓','winner'],
    ['+MMD / +sep-head','ESM-2','—','—','exp17 (running)'],
    ['u_gate / eps_mix','ESM-2','✗','✓','exp19 (running)'],
],[1.9,1.2,1.1,1.3,1.7],fs=10.5,hfs=10.5)
label(s,0.4,5.0,7,'Non-learned baselines',color=DARKBLUE,size=14)
bullets(s,0.4,5.4,7.3,1.6,[
    'context_mean (Δ=0), perturb_mean (drift), additive_obs (Σ observed singles; combos),',
    'corr (gene–gene control correlation), coexp (k-NN of training-TF Δ in control-feature space).',
],size=11)

box(s,8.0,1.0,5.0,6.0,fill=LIGHT,line=BLUE)
label(s,8.2,1.1,4.6,'Losses',color=BLUE)
eq_block(s,8.05,1.55,'loss_total',[
    r'$\mathcal{L}=\mathrm{MSE}(x,\hat x)+\beta\,\mathrm{KL}+\lambda_{p}\mathcal{L}_{\mathrm{pert}}+\mathcal{L}_{\mathrm{DAG}}$',
],width=4.9,fontsize=14)
text(s,8.2,2.25,4.6,0.3,'Perturbation (cascade) loss:',size=12,bold=True,color=DARK)
eq_block(s,8.05,2.55,'loss_pert',[
    r'$\mathcal{L}_{\mathrm{pert}}=\mathrm{MSE}(\widehat{\Delta x},\Delta x)$',
    r'$\;\;+\,w_{\mathrm{mmd}}\,\mathrm{MMD}\;+\;w_{c}\,\mathcal{L}_{\mathrm{InfoNCE}}$',
],width=4.9,fontsize=14)
text(s,8.2,3.6,4.6,0.3,'Drift-removed residual (specificity):',size=12,bold=True,color=DARK)
eq_block(s,8.05,3.9,'loss_drift',[
    r'$\mathrm{drift}=\frac{1}{|T|}\sum_{t\in T}\Delta_t,\;\; \mathrm{res}_t=\Delta_t-\mathrm{drift}$',
],width=4.9,fontsize=14)
text(s,8.2,4.6,4.6,0.3,'Contrastive (InfoNCE, drift-removed):',size=12,bold=True,color=DARK)
eq_block(s,8.05,4.9,'loss_nce',[
    r'$\mathcal{L}_{\mathrm{InfoNCE}}=-\log\frac{e^{\cos(\hat r_i,r_i)/\tau}}{\sum_j e^{\cos(\hat r_i,r_j)/\tau}}$',
],width=4.9,fontsize=14)
text(s,8.2,5.75,4.6,1.1,'InfoNCE makes each TF\'s predicted shift retrieve its OWN observed (drift-removed) shift — '
     'the specificity baselines cannot fake.',size=11,color=DARK)

# ════════════════════════ SLIDE 6 — Results: singles ════════════════════════
s=slide_blank(); title_bar(s,'3 · Results — Held-out Single TFs (zero-shot)')
fig_or_placeholder(s,0.4,1.05,8.0,5.0,'__PLACEHOLDER__singles',
                   'experiments/figures/joung/exp17_singles_panels.png')
box(s,8.7,1.05,4.3,5.4,fill=LIGHT,line=BLUE)
label(s,8.9,1.15,3.9,'Takeaways',color=BLUE)
bullets(s,8.9,1.6,3.9,4.6,[
    'Raw Pearson / Spearman / DEG-AUROC are SATURATED by perturb_mean (drift) and coexp — not where the model adds value.',
    'Model wins only on DRIFT-REMOVED Pearson + discrimination (the load-bearing, baseline-uncheatable metrics).',
    'esm_con (ESM input + contrastive) is the winner: disc-resid 0.615 vs 0.50 chance; beats even the coexpression prior.',
    'Residual-TARGET runs collapse (decoder conflict).',
],size=12)

# ════════════════════════ SLIDE 7 — Results: combos ════════════════════════
s=slide_blank(); title_bar(s,'3 · Results — Combo / Triple Composition')
fig_or_placeholder(s,0.35,1.05,6.3,4.0,'__PLACEHOLDER__combos',
                   'experiments/figures/joung/exp17_combos_panels.png')
fig_or_placeholder(s,7.0,1.05,6.0,4.0,'__PLACEHOLDER__pairs',
                   'experiments/figures/joung/exp17_pairs_vs_triples.png')
box(s,0.35,5.35,12.65,1.7,fill=LIGHT,line=AMBER)
label(s,0.55,5.45,12,'Takeaways',color=AMBER)
bullets(s,0.55,5.9,12.3,1.0,[
    'do-op (mutilation) beats zero-shot additive by a small margin — but LOSES to the additive_obs oracle (Σ observed singles): 0.25 vs 0.13.',
    'Only the contrastive models clear the perturb_mean floor on combos; no model achieves combo-level discrimination (≈ chance).',
],size=12)

# ════════════════════════ SLIDE 8 — Epistasis ════════════════════════
s=slide_blank(); title_bar(s,'4 · Epistasis — Is the do-operator Capturing Synergy?')
fig_or_placeholder(s,0.4,1.05,8.3,4.3,'__PLACEHOLDER__epi',
                   'experiments/figures/joung/exp17_epistasis_quartiles.png')
box(s,9.0,1.05,4.0,5.5,fill=LIGHT,line=RED)
label(s,9.2,1.15,3.6,'Verdict: epistasis-neutral',color=RED)
bullets(s,9.2,1.6,3.6,4.8,[
    'vs the model\'s OWN additive, the do-op adds only ≤0.05 Pearson — and that edge does NOT grow with epistasis (shrinks for esm_con).',
    'The apparent "gap closes on epistatic combos vs additive_obs" is the ORACLE degrading, not the model improving.',
    'Consistent with exp14 (Norman): the operator is epistasis-NEUTRAL, not epistasis-capturing.',
    'Singles-only training cannot learn interaction — needs combo data + interaction capacity.',
],size=11)

# ════════════════════════ SLIDE 9 — Next steps / architectures ════════════════════════
s=slide_blank(); title_bar(s,'5 · Next Steps — Architectures We\'re Pursuing')
label(s,0.4,0.95,12.5,'Singles-trainable non-additivity (decoder stays linear → GRN/CollecTRI intact)',color=DARK,size=15)
table(s,0.35,1.35,12.6,[
    ['direction','what it changes','why / hypothesis','status'],
    ['u_gate = MLP(S)','cross-program nonlinear gate on the do-op strength u','only singles-trainable non-additivity; gate sees multi-program single shifts','exp19 — training'],
    ['nonlinear ε-mix','nonlinear cross-program exogenous mixing','same goal via the ε-combination; zero-init → starts at linear baseline','exp19 — training'],
    ['u_gate + ε-mix','both','max singles-only non-additivity capacity','exp19 — training'],
    ['separate residual head','decouple a pert decoder from the GRN decoder','test residual target without the decoder conflict; A still gets gradients','exp17 — training'],
    ['re-add MMD','distributional set-matching term (exp14 recipe)','exp17 dropped MMD; restores perturbed-distribution shape','exp17 — training'],
    ['human-fetal annotation','swap mouse→human reference','more confident fate labels → enable fate-reward / RL identifiability','done (this deck)'],
    ['combo-training','train on a subset of combos (Norman has ~107; Joung only 44)','the ONLY lever that truly unlocks epistasis','planned (Norman)'],
],[2.4,3.6,4.6,1.8],fs=10,hfs=10.5)
box(s,0.35,6.35,12.65,0.85,fill=RGBColor(0xEC,0xF1,0xFB),line=BLUE)
text(s,0.55,6.5,12.3,0.6,'RL readiness: clears the discrimination gate over trivial baselines (single-step RL OK); '
     'combinatorial planning needs epistasis → combo data + interaction capacity. additive_obs is an oracle ceiling, not a gate.',size=12,color=DARK)

prs.save(OUT); print('Wrote',OUT,'(',len(prs.slides._sldIdLst),'slides )')
