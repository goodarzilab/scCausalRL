"""exp20 load-bearing-A investigation deck — same helper style as make_exp17_19_slides.py
(python-pptx + LaTeX→PNG via matplotlib mathtext). Figures from experiments/exp20_load_bearing/docs/figures/.
Run: python slides/make_exp20_slides.py  →  slides/exp20_load_bearing.pptx
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'exp20_load_bearing.pptx')
EQ_DIR = os.path.join(HERE, '_eq_cache_x20'); os.makedirs(EQ_DIR, exist_ok=True)
FIG = os.path.join(ROOT, 'experiments', 'exp20_load_bearing', 'docs', 'figures')

WHITE=RGBColor(0xFF,0xFF,0xFF); BLACK=RGBColor(0,0,0); DARK=RGBColor(0x1A,0x1A,0x2E)
BLUE=RGBColor(0x21,0x71,0xB5); GRAY=RGBColor(0x66,0x66,0x66); LIGHT=RGBColor(0xF5,0xF5,0xF5)
RED=RGBColor(0xC0,0x39,0x2B); GREEN=RGBColor(0x27,0xAE,0x60); AMBER=RGBColor(0xD9,0x82,0x1E)
DARKBLUE=RGBColor(0x0B,0x3D,0x91); PURPLE=RGBColor(0x6B,0x3F,0xA0)

prs = Presentation(); prs.slide_width=Inches(13.333); prs.slide_height=Inches(7.5)

def slide_blank():
    s=prs.slides.add_slide(prs.slide_layouts[6]); bg=s.background; bg.fill.solid(); bg.fill.fore_color.rgb=WHITE; return s
def text(s,l,t,w,h,txt,*,size=18,bold=False,color=BLACK,align=PP_ALIGN.LEFT,italic=False):
    tb=s.shapes.add_textbox(Inches(l),Inches(t),Inches(w),Inches(h)); tf=tb.text_frame; tf.word_wrap=True
    p=tf.paragraphs[0]; p.text=txt; p.font.size=Pt(size); p.font.bold=bold; p.font.italic=italic
    p.font.color.rgb=color; p.font.name='Calibri'; p.alignment=align; return tb
def bullets(s,l,t,w,h,items,*,size=14,color=BLACK,spacing=6):
    tb=s.shapes.add_textbox(Inches(l),Inches(t),Inches(w),Inches(h)); tf=tb.text_frame; tf.word_wrap=True
    for i,it in enumerate(items):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
        p.text=('• '+it) if not it.startswith('   ') else it
        p.font.size=Pt(size); p.font.color.rgb=color; p.font.name='Calibri'; p.space_after=Pt(spacing)
    return tb
def title_bar(s,txt,*,color=DARK):
    sh=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(0),Inches(0),Inches(13.333),Inches(0.85))
    sh.fill.solid(); sh.fill.fore_color.rgb=color; sh.line.fill.background(); sh.shadow.inherit=False
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
def render_eq(name,lines,*,fontsize=18,color='#0B3D91',dpi=300,ls=1.5,pad=0.22):
    path=os.path.join(EQ_DIR,f'{name}.png'); fig=plt.figure(figsize=(0.01,0.01)); fig.patch.set_alpha(0)
    ax=fig.add_axes([0,0,1,1]); ax.set_axis_off(); n=len(lines); ts=[]
    for i,ln in enumerate(lines):
        ts.append(ax.text(0.5,1.0-(i+0.5)/n,ln,fontsize=fontsize,color=color,ha='center',va='center'))
    fig.canvas.draw(); mw=max(t.get_window_extent().width for t in ts)/fig.dpi
    h=n*(fontsize/72.0)*ls; fig.set_size_inches(mw+2*pad,h+2*pad)
    fig.savefig(path,dpi=dpi,transparent=True,bbox_inches='tight',pad_inches=pad); plt.close(fig); return path
def eq(s,l,t,name,lines,*,width=None,fontsize=18,color='#0B3D91',ls=1.5):
    p=render_eq(name,lines,fontsize=fontsize,color=color,ls=ls)
    if width: s.shapes.add_picture(p,Inches(l),Inches(t),width=Inches(width))
    else: s.shapes.add_picture(p,Inches(l),Inches(t))
def figim(s,l,t,h,path,caption='',w=None):
    if os.path.exists(path):
        if w: s.shapes.add_picture(path,Inches(l),Inches(t),width=Inches(w))
        else: s.shapes.add_picture(path,Inches(l),Inches(t),height=Inches(h))
    if caption: text(s,l,t+h+0.02,(w or 8),0.35,caption,size=10,italic=True,color=GRAY,align=PP_ALIGN.CENTER)
def table(s,l,t,w,rows,col_w,*,header=True,fs=11,hfs=11,hcolor=DARKBLUE):
    nr=len(rows); nc=len(rows[0])
    gt=s.shapes.add_table(nr,nc,Inches(l),Inches(t),Inches(w),Inches(0.3*nr)).table
    for j,cw in enumerate(col_w): gt.columns[j].width=Inches(cw)
    for i,row in enumerate(rows):
        for j,val in enumerate(row):
            c=gt.cell(i,j); c.text=str(val); p=c.text_frame.paragraphs[0]
            p.font.size=Pt(hfs if (header and i==0) else fs); p.font.name='Calibri'
            p.font.bold=(header and i==0); p.font.color.rgb=WHITE if (header and i==0) else BLACK
            c.fill.solid(); c.fill.fore_color.rgb=hcolor if (header and i==0) else (LIGHT if i%2 else WHITE)
            c.margin_top=Pt(1); c.margin_bottom=Pt(1)
    return gt
def foot(s,txt):
    text(s,0.4,7.06,12.5,0.35,txt,size=10,italic=True,color=GRAY)

# ═══ 1 · Title ═══
s=slide_blank()
box(s,0,0,13.333,7.5,fill=DARK)
text(s,1,2.0,11.3,1.5,'Is the Latent DAG Load-Bearing?',size=40,bold=True,color=WHITE,align=PP_ALIGN.CENTER)
text(s,1,3.3,11.3,1.0,'An architectural + objective + identifiability investigation on the Joung TF atlas',
     size=20,color=RGBColor(0xC9,0xD3,0xE0),align=PP_ALIGN.CENTER)
text(s,1,5.5,11.3,0.5,'scCausalRL · exp20_load_bearing · shift-encoder causal-VAE world model',
     size=14,color=RGBColor(0x8A,0x93,0xA8),align=PP_ALIGN.CENTER)

# ═══ 2 · The model & the question ═══
s=slide_blank(); title_bar(s,'2 · The model & the two questions')
label(s,0.5,1.05,7,'Shift-encoder causal-VAE (per-cell) + do-operator')
eq(s,0.6,1.6,'m_enc',[r'$x \;\to\; \varepsilon \;\to\; z=(I-A^{\top})^{-1}\varepsilon \;\to\; \hat{x}=Dz$'],width=7.6,fontsize=17)
text(s,0.7,2.12,7.4,0.35,'(encoder)            (linear-Gaussian SCM)            (linear decoder)',size=11,color=GRAY)
label(s,0.5,2.6,7,'Perturbation = mutilated do-operator (shift S from a per-TF ESM descriptor)')
eq(s,0.6,3.1,'m_do',[r'$\tilde\varepsilon=\varepsilon_{ctrl}+S,\qquad z_{post}=(I-\mathrm{diag}(1-u)\,A^{\top})^{-1}\tilde\varepsilon,\qquad u=\mathrm{clip}(\beta|S|)$'],width=9.6,fontsize=16)
label(s,0.5,4.1,7,'GRN read out from the latent DAG')
eq(s,0.6,4.55,'m_grn',[r'$G = D\,A\,D^{\top}$'],width=2.1,fontsize=17)
box(s,7.2,1.5,5.7,2.0,fill=RGBColor(0xEC,0xF1,0xFB),line=BLUE)
label(s,7.45,1.62,5.2,'Q1 — Causality',color=BLUE)
text(s,7.45,2.05,5.2,1.3,'Is A load-bearing? Does the learned latent DAG carry prediction, '
     'and is the projected GRN biologically real?',size=13,color=BLACK)
box(s,7.2,3.7,5.7,2.0,fill=RGBColor(0xFB,0xEC,0xEC),line=RED)
label(s,7.45,3.82,5.2,'Q2 — Prediction',color=RED)
text(s,7.45,4.25,5.2,1.3,'Can it predict a HELD-OUT TF’s effect zero-shot (unseen perturbation), '
     'from protein sequence alone?',size=13,color=BLACK)
box(s,0.5,5.3,12.4,1.5,fill=RGBColor(0xF5,0xF5,0xEA),line=AMBER)
text(s,0.75,5.45,12,1.2,'Prior finding (27-run exp19 sweep): A is INERT and discrimination is WEAK. exp20 attacks both at the '
     'ARCHITECTURE + OBJECTIVE level (not loss tuning), retires the CollecTRI prior, and closes with a synthetic '
     'ground-truth identifiability test.',size=14,bold=True,color=DARK)
foot(s,'Eval is always ZERO-SHOT on held-out TF genes (gene-level split); metrics are drift-removed.')

# ═══ 3 · The two failure modes / escape hatches ═══
s=slide_blank(); title_bar(s,'3 · Two failure modes — and the escape hatches that cause them')
box(s,0.4,1.1,6.15,5.4,fill=RGBColor(0xEC,0xF1,0xFB),line=BLUE)
label(s,0.7,1.25,5.6,'A inert  (identifiability)')
bullets(s,0.7,1.75,5.6,4.6,[
 'A=0 stays optimal: a per-cell ε reconstructs each cell, so A carries nothing.',
 'DECODER ABSORPTION — the linear decoder can absorb the SCM operator:',
 '   (see equation) → for ANY A there is a D giving identical predictions.',
 'A dense learned shift reproduces the propagated effect directly ⇒ A never needed.',
 'Consequence: A_gap ≈ 0; the "GRN" is decoder co-expression, not causal.'],size=13.5)
eq(s,0.7,4.35,'absorb',[r'$D_{\mathrm{eff}} = D\,(I-\mathrm{diag}(1-u)A^{\top})^{-1}$'],width=5.2,fontsize=15,color='#0B3D91')
box(s,6.75,1.1,6.15,5.4,fill=RGBColor(0xFB,0xEC,0xEC),line=RED)
label(s,7.05,1.25,5.6,'Discrimination weak  (mode collapse)')
bullets(s,7.05,1.75,5.6,4.6,[
 'Joung Δ is ~99% a shared differentiation program (one dominant mode).',
 'The model predicts that shared program scaled by a single magnitude scalar.',
 'Per-TF predictions become ~identical  →  crossTF_div ≈ 0.02.',
 'Zero-shot on unseen TFs: the ESM→effect map does not transfer at ~5% SNR.',
 'Consequence: discrimination pinned near chance (0.5).'],size=13.5)
foot(s,'These are SEPARATE problems: #1 is architecture/identifiability; #2 is objective + data SNR.')

# ═══ 4 · The changes we ran (ablation catalog + run names) ═══
s=slide_blank(); title_bar(s,'4 · What we changed — ablations & the runs (all PRIOR-OFF)')
rows=[['Fix (ingredient)','What it changes','Targets'],
 ['Fix 1 · structural do-op','single mutilated solve, S into exogenous noise (no z-target leak)','A inert'],
 ['Fix 2 · sparse L0 gate','2nd ESM head gates programs; L0 on gate existence (soft to hard)','A inert'],
 ['frozen GO+PCA decoder','D fixed to GO:BP membership + PCA cols (requires_grad=False)','absorption'],
 ['whitened pert target','MSE on drift-removed + per-gene z-scored residual','mode collapse'],
 ['pseudobulk target','per-TF population-mean delta instead of noisy per-cell mean','SNR / collapse'],
 ['project-out shared dir','remove the drift-direction component of the error','shared handling'],
 ['staged 3-phase L0','decouple gate-closing from A-learning (A competent first)','gate-vs-A race'],
 ['nonlinear SCM','z=tanh(A^T z)+eps via Picard; Jacobian GRN','A expressiveness'],
 ['hard self-gate','shift restricted to gene g own top-k decoder programs','localization'],
 ['multi-context (+Norman)','2nd cell context via modality token; ESM bridge; shared held-out','generalization']]
table(s,0.4,1.0,12.55,rows,[2.55,7.1,2.9],fs=11.5,hfs=12)
box(s,0.4,4.8,12.55,2.05,fill=RGBColor(0xF3,0xF6,0xFC),line=BLUE)
label(s,0.6,4.88,12,'The runs  (recipe -> name)',color=DARKBLUE,size=15)
bullets(s,0.6,5.3,12.3,1.5,[
 'Base arms (grid x20po):  min_esm = Fix1 only  .  sparse_esm = +Fix2 gate  .  sparse_con = +InfoNCE contrastive  .  '
 'sparse_resid = +drift-removed target  .  sparse_pertonly = +down-weighted recon  .  sparse_residrecon = +residual recon  .  sparse_all = all',
 'Objective sweep (grid x20R, on sparse_con base):  R1 per-example  .  R2 pseudobulk  .  R3 project-out  .  '
 'R4 pseudobulk+project  .  R5 = R4 + contrastive x10  .  R6 = R4 + energy/MMD',
 'Architecture (x20arch / x20frz):  staged_esm, staged_con (staged L0)  .  whiten  .  nonlinSCM  .  frozenD64 (+nonlin / +whiten)   '
 '.   In flight: J1/J2 (hard self-gate), JN0/JN1 (Joung+Norman)'],size=11.5,color=BLACK,spacing=5)
foot(s,'Grids: x20po (prior-off base) . x20R (objective) . x20arch/x20frz (architecture) . synthetic (identifiability). Prior-ON x20 kept only as the GRN memorization control.')

# ═══ 5 · Metrics & methodology ═══
s=slide_blank(); title_bar(s,'5 · Metrics & methodology (one eval script for every arm)')
label(s,0.5,1.05,12,'Prediction — zero-shot held-out TFs, drift-removed')
bullets(s,0.6,1.5,12,1.3,[
 'disc_resid = drift-removed discrimination_cos (does pred_i retrieve obs_i vs other TFs; chance 0.5)',
 'train vs held-out disc = overfitting / generalization gap'],size=13.5)
label(s,0.5,2.9,12,'Causality — A load-bearing (real vs zero vs shuffled A)')
eq(s,0.6,3.35,'agap',[r'$S\_A_{gap}=\mathrm{spear}_{\mathrm{real}\,A}-\max(\mathrm{spear}_{A=0},\ \mathrm{spear}_{\mathrm{shuf}\,A})\;>0\ \Rightarrow\ A\ \mathrm{helps}$'],width=8.6,fontsize=15)
bullets(s,0.6,4.1,12,1.3,[
 'probe_A = 1−cos(Δ, Δ|A=0)  — does removing A change the output (A used?)',
 'probe_Dperm = 1−cos(Δ, Δ|D shuffled)  — decoder dominance;   crossTF_div — pred diversity (mode collapse)'],size=13.5)
label(s,0.5,5.4,12,'Biology (prior-free) — projected GRN vs known networks, with shuffle-A / random-D controls')
bullets(s,0.6,5.85,12,0.9,[
 'CollecTRI / DoRothEA AUROC  +  fetal-atlas fate-hierarchy AUROC (same-fate TF pairs)'],size=13.5)
foot(s,'All arms evaluated on the FINAL epoch-60 checkpoint (not best-by-val-loss).')

# ═══ 6 · Metric definitions ═══
s=slide_blank(); title_bar(s,'6 · Metric definitions & chance levels')
text(s,0.5,0.95,12.4,0.4,'c(a,b) = mean-centered cosine.   r̂=pred−drift, r=obs−drift.   Δ̂(A)=do-op prediction under adjacency A  (A=0 zeroed, Aπ column-shuffled).',size=11,italic=True,color=GRAY)
label(s,0.5,1.45,6,'Prediction   (chance 0.5)',color=RED)
eq(s,0.6,1.9,'d_disc',[r'$\mathrm{disc\_resid}=\frac{1}{T}\sum_i\frac{1}{T-1}\sum_{j\neq i}\mathbf{1}\,[\,c(\hat{r}_i,r_i)>c(\hat{r}_i,r_j)\,]$'],width=6.1,fontsize=13)
eq(s,0.6,2.7,'d_pear',[r'$\mathrm{pear\_resid}=\frac{1}{T}\sum_i \mathrm{corr}(\hat{r}_i,\,r_i)$'],width=3.5,fontsize=13)
label(s,0.5,3.45,6,'Causality — is A load-bearing?',color=BLUE)
eq(s,0.6,3.9,'d_agap',[r'$S\_A_{gap}=\rho(A_{\mathrm{real}})-\max(\rho(0),\,\rho(A_{\pi}))$',
                       r'$\rho(A)=\frac{1}{T}\sum_i \mathrm{spearman}_{g\in DEG_i}(|\hat{\Delta}_i(A)_g-d_g|,\,|\Delta_{i,g}-d_g|)$'],width=6.7,fontsize=12)
eq(s,0.6,5.0,'d_probeA',[r'$\mathrm{probe\_A}=\frac{1}{T}\sum_i[\,1-c(\hat{\Delta}_i(A_{\mathrm{real}}),\,\hat{\Delta}_i(0))\,]$'],width=5.6,fontsize=13)
label(s,7.05,1.45,6,'Mechanism',color=PURPLE)
eq(s,7.15,1.9,'d_dperm',[r'$\mathrm{probe\_Dperm}=\frac{1}{T}\sum_i[\,1-c(\hat{\Delta}_i(D),\,\hat{\Delta}_i(D_{\pi}))\,]$'],width=5.7,fontsize=12.5)
eq(s,7.15,2.7,'d_cross',[r'$\mathrm{crossTF\_div}=\mathrm{mean}_{i<j}[\,1-c(\hat{\Delta}_i,\hat{\Delta}_j)\,]$'],width=5.2,fontsize=12.5)
label(s,7.05,3.45,6,'Biology  (AUROC, prior-free)',color=GREEN)
eq(s,7.15,3.9,'d_grn',[r'$\mathrm{collectri}=\mathrm{AUROC}(\mathbf{1}[tf\!\rightarrow\!g],\ |D_{tf}AD^{\top}|)$'],width=5.9,fontsize=12)
eq(s,7.15,4.5,'d_fate',[r'$\mathrm{fate}=\mathrm{AUROC}(\mathbf{1}[\mathrm{fate}_i{=}\mathrm{fate}_j],\ |G_{ij}|)$'],width=5.6,fontsize=12)
eq(s,7.15,5.1,'d_graph',[r'$\mathrm{graph}=\mathrm{AUROC}(\mathbf{1}[|A^{*}_{ij}|{>}0],\ |A^{\mathrm{learn}}_{ij}|)$'],width=5.6,fontsize=12)
box(s,0.5,5.95,12.4,1.05,fill=RGBColor(0xF5,0xF5,0xEA),line=AMBER)
text(s,0.72,6.05,12.0,0.95,'KEY DISTINCTION: probe_A = “A is USED” (output changes if A removed);  S_Agap = “A is USEFUL” (drift-removed prediction improves). '
     'Finding: probe_A can be 0.8 while S_Agap≈0 — used, not useful.   Note: grn_dag_sweep’s “disc” column = retrieval-top-1 (chance 1/T≈0.006), NOT disc_resid (chance 0.5).',size=11.5,bold=True,color=DARK)

# ═══ 7 · FIG 1 prediction ladder ═══
s=slide_blank(); title_bar(s,'7 · Perturbation prediction — nothing beats the prior-off ceiling')
figim(s,0.5,1.05,5.9,os.path.join(FIG,'fig1_prediction_ladder.png'))
bullets(s,8.4,1.5,4.6,5,[
 'Best = prior-off sparse_con 0.639 ≈ exp19 epsmix ceiling 0.647.',
 'Turning the CollecTRI PRIOR OFF is the only change that helped (+0.05).',
 'whiten / resid / all / frozen arms COLLAPSE to chance (0.50).',
 'staged-L0 and nonlinear-SCM are flat-to-worse.',
 'Net: no architectural/objective fix beat a simple prior-off shift+decoder.'],size=14,color=DARK)
foot(s,'Held-out zero-shot TFs; disc_resid, chance 0.5, exp19 ceiling 0.647.')

# ═══ 8 · FIG 2 causality ═══
s=slide_blank(); title_bar(s,'8 · Causality — A is not load-bearing')
figim(s,0.5,1.1,4.9,os.path.join(FIG,'fig2_causality_Agap.png'),w=12.3)
bullets(s,0.6,6.2,12.4,1.0,[
 'real-A ≈ zero-A ≈ shuffled-A on drift-removed prediction; A_gap ≈ 0 for every prior-off arm (±0.05). '
 'Removing or shuffling A barely changes the match — A carries no predictive causal structure.'],size=13.5,color=DARK)

# ═══ 9 · FIG 3 mechanism ═══
s=slide_blank(); title_bar(s,'9 · Mechanism — decoder absorbs the operator; predictions collapse')
figim(s,0.5,1.05,5.4,os.path.join(FIG,'fig3_mechanism.png'))
label(s,9.0,1.3,4.1,'Why A is redundant')
eq(s,9.0,1.8,'absorb2',[r'$\Delta = D\,\Delta z$'],width=2.0,fontsize=16)
bullets(s,9.0,2.5,4.1,4.2,[
 'probe_Dperm ≈ 1.0: shuffling D destroys the output → D carries the gene pattern.',
 'probe_A ≤ 0.16 (soft gate): A a minor add-on.',
 'crossTF_div ≈ 0.02: ~99% identical predictions across TFs = mode collapse.',
 'A single scalar shift_scale absorbs 27–47% of the pert-loss gradient.'],size=13,color=DARK)
foot(s,'The shift is load-bearing for MAGNITUDE, not identity; the decoder carries the gene-space pattern.')

# ═══ 10 · FIG 4 biology ═══
s=slide_blank(); title_bar(s,'10 · Biology — the projected GRN is prior memorization')
figim(s,0.5,1.05,5.4,os.path.join(FIG,'fig4_biology_grn.png'))
bullets(s,9.0,1.5,4.1,5,[
 'CollecTRI AUROC 0.98 (prior ON) → 0.57 (prior OFF) = co-expression floor.',
 'Shuffling A does NOT drop it → the signal lives in the decoder (D·Dᵀ), not A.',
 'Fetal-atlas FATE hierarchy: AUROC 0.51 = chance → no developmental structure.',
 'random-D → chance confirms D carries the (co-expression) signal.'],size=13.5,color=DARK)
foot(s,'Prior-free GRN validation: neither TF→target edges nor developmental hierarchy are recovered from data.')

# ═══ 11 · FIG 5 R-sweep ═══
s=slide_blank(); title_bar(s,'11 · Objective sweep — break mode collapse, but not generalization')
eq(s,0.5,1.0,'decomp',[r'$\Delta_{\mathrm{pred}}(p)=\;o\;(\mathrm{shared\ offset})\;+\;f(\mathrm{shift}(p),A)\;(\mathrm{TF\text{-}specific})$'],width=8.6,fontsize=15)
text(s,0.55,1.75,9.5,0.4,'Lever A: pseudobulk per-TF target (SNR).   Lever B: project the shared direction out of the loss.',size=13,color=GRAY)
figim(s,0.5,2.2,4.5,os.path.join(FIG,'fig5_Rsweep.png'),w=12.3)
bullets(s,0.6,6.85,12.4,0.8,[
 'Pseudobulk (Lever A) lifts crossTF_div 0.03→0.2 and probe_A→0.9 (A engaged) — but disc stays at chance: '
 'diverse-but-WRONG. Project-out alone (R3) does nothing. The binding constraint is zero-shot generalization.'],size=12.5,color=DARK)

# ═══ 12 · FIG 6 train/test gap ═══
s=slide_blank(); title_bar(s,'12 · Generalization is the wall — memorize seen, fail unseen')
figim(s,0.5,1.15,5.5,os.path.join(FIG,'fig6_train_vs_heldout.png'))
bullets(s,8.4,1.6,4.6,5,[
 'sparse_con: TRAIN disc 0.93 → held-out 0.62 (+0.31 gap).',
 'On seen TFs the model discriminates almost perfectly; it just does not transfer.',
 'Shift ‖S‖ stays full-magnitude on unseen TFs → confidently WRONG direction (not mean-regression).',
 'pseudobulk arms fit train worse (0.56) and still fail held-out.'],size=14,color=DARK)
foot(s,'The failure is zero-shot generalization of the TF→effect map — not fitting capacity.')

# ═══ 13 · FIG 7 synthetic ═══
s=slide_blank(); title_bar(s,'13 · Synthetic control — method sound, A non-identifiable')
eq(s,0.5,1.0,'syn',[r'$S^{*}_{g}=m_g\odot(W^{*}d_g),\qquad x=D^{*}z+\sigma\eta$'],width=6.0,fontsize=15)
text(s,6.8,1.15,6.2,0.4,'planted A*, sparse shift LINEAR in the descriptor (generalizable)',size=12,italic=True,color=GRAY)
figim(s,0.5,1.9,4.35,os.path.join(FIG,'fig7_synthetic.png'),w=12.3)
bullets(s,0.6,6.5,12.4,0.9,[
 'Planted A*/D*/generalizable shift: the shift-encoder GENERALIZES (disc 0.66; perturbation-blind localization → 0.90) '
 'but graph-recovery AUROC stays at CHANCE (~0.5) — A is non-identifiable (shift+decoder are a sufficient statistic).'],size=12.5,color=DARK)

# ═══ 14 · Master results table ═══
s=slide_blank(); title_bar(s,'14 · Master results (representative arms, prior-off)')
rows=[['arm','fix','disc_resid','S_Agap','crossTF','verdict'],
 ['sparse_con','baseline (best)','0.639','−0.05','0.02','ceiling; A idle'],
 ['sparse_pertonly','baseline','0.621','−0.03','0.02','—'],
 ['staged_con','staged L0','0.611','−0.05','—','no gain'],
 ['nonlinSCM','nonlinear SCM','0.570','−0.08','—','worse; A anti-useful'],
 ['staged_esm','staged L0 (no-con)','0.553','+0.05','—','no rescue'],
 ['R4','pseudobulk+proj','0.502','+0.03','0.20','collapse; crossTF↑ probeA↑'],
 ['whiten','whitened target','0.500','+0.06','—','collapse to chance'],
 ['frozenD64','frozen decoder','0.501','+0.00','0.02','collapse (basis span)'],
 ['frozenD64_nonlin','frozen + nonlinear','0.500','+0.01','0.00','collapse'],
]
table(s,0.4,1.05,12.55,rows,[2.2,2.9,1.7,1.4,1.35,3.0],fs=12.5,hfs=13,hcolor=DARK)
box(s,0.4,5.15,12.55,1.6,fill=RGBColor(0xF5,0xF5,0xEA),line=AMBER)
text(s,0.65,5.3,12.1,1.4,'Reference: exp19 best (epsmix, prior-on) disc_resid 0.647 · Joung split-half discriminability ceiling ≈ 5% · '
     'CollecTRI prior-on 0.97 → prior-off 0.57 (memorization). A is used-but-not-useful wherever the gate opens; '
     'no arm both discriminates AND uses A.',size=13,bold=True,color=DARK)
foot(s,'Full 22-arm table in docs/MASTER_ALL.md; figures in docs/figures/.')

# ═══ 15 · Verdict & next levers ═══
s=slide_blank(); title_bar(s,'15 · Verdict & the two real levers')
box(s,0.4,1.1,12.5,1.7,fill=RGBColor(0xEC,0xF7,0xEC),line=GREEN)
text(s,0.65,1.25,12.0,1.5,'CONCLUSION — The architecture is sound (it generalizes on ideal synthetic data). On Joung the binding '
     'constraint is ZERO-SHOT GENERALIZATION of the TF→effect map (models memorize seen TFs, disc 0.93→0.62). '
     'And A is NON-IDENTIFIABLE / redundant in this shift-based class — even with a planted graph — because the '
     'shift + decoder are a sufficient statistic.',size=14,bold=True,color=RGBColor(0x14,0x60,0x14))
label(s,0.5,3.1,12,'Two levers (neither is "more SCM machinery on Joung")')
box(s,0.4,3.55,6.15,2.7,fill=RGBColor(0xEC,0xF1,0xFB),line=BLUE)
label(s,0.7,3.7,5.6,'A · Multi-context (Norman)',color=BLUE)
bullets(s,0.7,4.2,5.6,2.0,[
 'Add K562 CRISPRa as a 2nd context (modality token + ESM bridge).',
 'Lets the model separate context-shared from TF-specific signal.',
 'Directly targets the generalization ceiling. (JN0/JN1 in flight.)'],size=13)
box(s,6.75,3.55,6.15,2.7,fill=RGBColor(0xF3,0xEC,0xFB),line=PURPLE)
label(s,7.05,3.7,5.6,'B · Different identification',color=PURPLE)
bullets(s,7.05,4.2,5.6,2.0,[
 'Load-bearing A needs a setup where the shift CANNOT express propagation.',
 'e.g. fit A to the interventional DISTRIBUTION (discrepancy-VAE).',
 'Hard self-gate localization + frozen decoder tested; still redundant.'],size=13)
foot(s,'In flight: J1/J2 (hard self-gate localization) + JN0/JN1 (Joung+Norman multi-context, 43 shared TFs).')

prs.save(OUT); print('saved', OUT, '|', len(prs.slides._sldIdLst), 'slides')
