"""exp20 lb9 sweep deck — fixing zero-shot generalization (MMD + DEG-MSE) + combo epistasis.
Same helper style as make_exp20_slides.py (python-pptx + LaTeX->PNG via matplotlib mathtext).
Figures from experiments/exp20_load_bearing/results/figures/.
Run: python slides/make_exp20_lb9_slides.py  ->  slides/exp20_lb9_generalization.pptx
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
OUT = os.path.join(HERE, 'exp20_lb9_generalization.pptx')
EQ_DIR = os.path.join(HERE, '_eq_cache_lb9'); os.makedirs(EQ_DIR, exist_ok=True)
FIG = os.path.join(ROOT, 'experiments', 'exp20_load_bearing', 'results', 'figures')

WHITE=RGBColor(0xFF,0xFF,0xFF); BLACK=RGBColor(0,0,0); DARK=RGBColor(0x1A,0x1A,0x2E)
BLUE=RGBColor(0x21,0x71,0xB5); GRAY=RGBColor(0x66,0x66,0x66); LIGHT=RGBColor(0xF5,0xF5,0xF5)
RED=RGBColor(0xC0,0x39,0x2B); GREEN=RGBColor(0x27,0xAE,0x60); AMBER=RGBColor(0xD9,0x82,0x1E)
DARKBLUE=RGBColor(0x0B,0x3D,0x91); PURPLE=RGBColor(0x6B,0x3F,0xA0); TEAL=RGBColor(0x1F,0x9E,0x89)

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
def render_eq(name,lines,*,fontsize=18,color='#0B3D91',dpi=300,ls=1.15,pad=0.06):
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
s=slide_blank(); box(s,0,0,13.333,7.5,fill=DARK)
text(s,1,2.0,11.3,1.5,'Fixing Zero-Shot Generalization',size=40,bold=True,color=WHITE,align=PP_ALIGN.CENTER)
text(s,1,3.25,11.3,1.0,'MMD + DE-gene loss beat the copy baseline; combo-training captures epistasis',
     size=20,color=RGBColor(0xC9,0xD3,0xE0),align=PP_ALIGN.CENTER)
text(s,1,5.5,11.3,0.5,'scCausalRL · exp20_load_bearing · lb9 sweep · Joung TF atlas (vocab-v2b), prior-off',
     size=14,color=RGBColor(0x8A,0x93,0xA8),align=PP_ALIGN.CENTER)

# ═══ 2 · Where we left off ═══
s=slide_blank(); title_bar(s,'2 · Where we left off — the wall was generalization')
box(s,0.4,1.1,12.5,1.6,fill=RGBColor(0xFB,0xEC,0xEC),line=RED)
text(s,0.65,1.25,12.0,1.4,'exp20 verdict: the architecture is sound, but on Joung the binding constraint is ZERO-SHOT '
     'GENERALIZATION. The model memorized seen TFs (train allDEG 0.74) and fell to ~0 on held-out TFs — '
     'BELOW a dumb ESM-nearest-neighbour COPY baseline (allDEG 0.31). disc_resid was flattering but not correctness.',
     size=14,bold=True,color=DARK)
label(s,0.5,3.0,12,'This deck — two questions')
box(s,0.4,3.5,6.15,3.0,fill=RGBColor(0xEC,0xF1,0xFB),line=BLUE)
label(s,0.7,3.65,5.6,'Q1 · Beat the copy?',color=BLUE)
bullets(s,0.7,4.15,5.6,2.2,['Can any objective/architecture change push held-out DE-gene prediction ABOVE the ESM-kNN copy?',
 'Levers: MMD distribution-matching, drift-removed DE-gene MSE, ESM-noise, nonlinear-SCM, free decoder, retrieval anchor.'],size=13.5)
box(s,6.75,3.5,6.15,3.0,fill=RGBColor(0xF3,0xEC,0xFB),line=PURPLE)
label(s,7.05,3.65,5.6,'Q2 · Capture epistasis?',color=PURPLE)
bullets(s,7.05,4.15,5.6,2.2,['Does the do-operator predict combinatorial (non-additive) TF effects — beyond simple additivity?',
 'Compare singles-only vs combo-trained (50/50 split), same held-out combos.'],size=13.5)
foot(s,'All eval is zero-shot on held-out TFs (gene-level split); metrics are drift-removed; prior OFF everywhere.')

# ═══ 3 · Vocab v2b ═══
s=slide_blank(); title_bar(s,'3 · The dataset — Joung TF atlas, cleaned to vocab-v2b')
label(s,0.5,1.05,12,'How it was cleaned')
bullets(s,0.6,1.5,7.2,2.6,[
 'Rebuilt from the RAW counts atlas (37,528 genes) — the legacy 9,923-gene proc overlapped the new vocab by only 61%.',
 'Gene axis = 4,000 Seurat-v3 highly-variable genes  ∪  curated fate markers  →  4,366 genes.',
 'TF vocab: isoform-merged to 661 perturbations, then filtered to EFFECTIVE TFs — 166 inert/absent perturbations dropped (no measurable response).',
 'Gene-level split of the 495 effective TFs → strict zero-shot held-out test set.'],size=13.5)
rows=[['','count','note'],
 ['Genes (v2b vocab)','4,366','4000 Seurat-v3 HVG ∪ fate markers'],
 ['TF perturbations','661','isoform-merged'],
 ['  → effective (kept)','495','166 inert/absent dropped'],
 ['  → train / val / test','396 / 50 / 49','zero-shot gene-level split'],
 ['Cells (total)','1,145,823',''],
 ['  → control (GFP/mCherry)','127,813','baseline for Δ']]
table(s,0.5,4.25,8.3,rows,[3.0,2.2,3.1],fs=12,hfs=12.5)
box(s,9.1,1.2,3.8,5.3,fill=RGBColor(0xF3,0xF6,0xFC),line=BLUE)
label(s,9.35,1.35,3.4,'Δ, drift, residual',color=DARKBLUE,size=15)
eq(s,9.4,2.05,'v_delta',[r'$\Delta_p = \bar{x}_{p}-\bar{x}_{ctrl}$'],width=2.9,fontsize=15)
text(s,9.4,2.62,3.4,0.4,'per-TF log-fold change',size=10.5,italic=True,color=GRAY)
eq(s,9.4,3.25,'v_drift',[r'$d=\frac{1}{|T_{tr}|}\sum_{p}\Delta_p$'],width=2.4,fontsize=15)
text(s,9.4,4.0,3.4,0.5,'shared differentiation drift (≈99% of signal)',size=10.5,italic=True,color=GRAY)
eq(s,9.4,4.75,'v_resid',[r'$r_p=\Delta_p-d$'],width=2.3,fontsize=15)
text(s,9.4,5.35,3.4,0.5,'TF-specific residual — what we score',size=10.5,italic=True,color=GRAY)
foot(s,'Same eval contract as exp17/19; load_joung lognorms raw counts downstream.')

# ═══ 4 · Metrics & why not disc ═══
s=slide_blank(); title_bar(s,'4 · Metrics — and why disc_resid alone is misleading')
text(s,0.5,0.95,12.4,0.4,'c(a,b) = mean-centered cosine.   r̂=pred−drift, r=obs−drift.   T = # held-out TFs (n=49).   KO = perturbed gene (excluded).',size=11,italic=True,color=GRAY)
label(s,0.5,1.5,6,'disc_resid  (TF discrimination, chance 0.5)',color=RED)
eq(s,0.6,2.0,'m_disc',[r'$\mathrm{disc}=\frac{1}{T}\sum_i\frac{1}{T-1}\sum_{j\neq i}\mathbf{1}[\,c(\hat{r}_i,r_i)>c(\hat{r}_i,r_j)\,]$'],width=5.6,fontsize=13)
label(s,0.5,3.05,6,'Pnk  (whole-profile Pearson, chance 0)',color=BLUE)
eq(s,0.6,3.55,'m_pnk',[r'$\mathrm{Pnk}=\frac{1}{T}\sum_i \mathrm{corr}_{g\neq KO}(\hat{r}_{i,g},\,r_{i,g})$'],width=4.4,fontsize=13)
label(s,0.5,4.6,6,'allDEG  (DE-gene Pearson, chance 0)',color=TEAL)
eq(s,0.6,5.1,'m_adeg',[r'$\mathrm{allDEG}=\frac{1}{T}\sum_i \mathrm{corr}_{g:\,|r_{i,g}|>0.1}(\hat{r}_{i,g},\,r_{i,g})$'],width=5.0,fontsize=13)
figim(s,6.85,1.2,4.7,os.path.join(FIG,'disc_vs_correctness.png'),w=6.15)
box(s,0.4,6.2,12.55,0.75,fill=RGBColor(0xF5,0xF5,0xEA),line=AMBER)
text(s,0.6,6.3,12.15,0.6,'WHY NOT disc alone: it rewards a TF fingerprint in low-magnitude BULK genes (≈ the training contrastive objective, circular) — '
     'across the sweep disc ANTI-correlates with correctness (right): the winner has LOW disc / HIGH allDEG; MMD=10 has HIGH disc / ~0 allDEG.',size=11.5,bold=True,color=DARK)

# ═══ 5 · The sweep ═══
s=slide_blank(); title_bar(s,'5 · The lb9 sweep — generalization levers (prior-off)')
rows=[['lever','what it adds','arms'],
 ['MMD (distribution match)','RBF-MMD(pred cells, obs cells) added to raw-Δ MSE','MMD ∈ {0.25,0.5,1,2,5,10}'],
 ['+ DE-gene MSE','drift-removed, DE-gene-masked, magnitude-preserving MSE','mmd1_deg2, mmd2_deg2'],
 ['+ ESM-noise / wd','jitter the ESM→shift map (anti-memorization) + weight-decay','mmd1_esmnoise(_wd)'],
 ['+ retrieval anchor','self-masked D†·residual memory blended into exogenous S','mmd1_retr_k20'],
 ['nonlinear SCM','z = φ(Aᵀz) + ε via Picard iteration','nlSCM_mmd1, nlSCM_mmd2_deg2'],
 ['free decoder','learnable linear D (vs GO-masked)','free357'],
 ['combo training','hold out TFs AND combos; do-op composes shifts','Jcombo / JNcombo (50/50)']]
table(s,0.4,1.05,12.55,rows,[2.9,6.35,3.3],fs=12,hfs=12.5)
box(s,0.4,3.7,12.55,1.15,fill=RGBColor(0xEC,0xF7,0xEC),line=GREEN)
text(s,0.65,3.82,12.0,1.05,'Backbone (best from before): masked-GO decoder + structural do-op + L0 target gate + ESM shift-encoder + MSE + InfoNCE, prior OFF. '
     'MMD is the proven lever; DE-gene MSE is the diagnosed-missing term (nothing rewarded drift-removed DE-gene magnitude).',size=13,bold=True,color=RGBColor(0x14,0x60,0x14))
label(s,0.5,5.25,12,'The do-operator (mutilated SCM solve)')
eq(s,0.6,5.8,'do_op',[r'$\tilde\varepsilon=\varepsilon_{ctrl}+S,\quad z_{post}=(I-\mathrm{diag}(1-u)A^{\top})^{-1}\tilde\varepsilon,\quad u=\mathrm{clip}(\beta|S|)$'],width=9.2,fontsize=15)
foot(s,'Every arm = one knob change on the same backbone; judged on held-out drift-removed allDEG / Pnk + real−shuffled A-gain.')

# ═══ 6 · Eval ranking ═══
s=slide_blank(); title_bar(s,'6 · Eval across all arms — the winner beats the copy  (n=49)')
figim(s,0.35,1.0,5.6,os.path.join(FIG,'lb9_ranking.png'),w=9.1)
box(s,9.65,1.1,3.35,5.4,fill=RGBColor(0xEC,0xF7,0xEC),line=GREEN)
label(s,9.85,1.25,3.0,'Result',color=RGBColor(0x14,0x60,0x14))
bullets(s,9.85,1.75,3.0,4.6,[
 'mmd1_deg2 BEATS ESM-kNN: allDEG 0.34 vs 0.31, Pnk 0.24 vs 0.21 (4000-ctrl, 3-seed).',
 'MMD optimum ≈ 1–2; 0.25 and 10 collapse.',
 'DE-gene MSE stacks (0.25→0.34).',
 'retrieval anchor helped here (not on old backbone).',
 'ESM-noise / free-decoder did NOT help.'],size=12.5,color=DARK)
foot(s,'Bars ±1 SEM over n=49 held-out TFs; red dashed = ESM-kNN copy; disc chance 0.5, correlation chance 0.')

# ═══ 7 · Best model vs baselines + A ═══
s=slide_blank(); title_bar(s,'7 · Best prior-off model vs baselines + A is load-bearing  (n=49)')
figim(s,0.5,1.2,5.1,os.path.join(FIG,'best_model_heldout.png'),w=12.3)
bullets(s,0.6,6.35,12.4,0.8,[
 'Left: the winner (MMD + DE-gene MSE) beats ESM-kNN / corr-blind / perturb-mean on Pnk & allDEG (disc lower — the gameable metric).   '
 'Right: real A beats SHUFFLED ≈ zero A by +0.39 Pnk / +0.72 allDEG — the learned wiring is genuinely load-bearing.'],size=13,color=DARK)

# ═══ 8 · Per-TF ═══
s=slide_blank(); title_bar(s,'8 · Per-TF prediction — broadly competitive with the copy  (n=49)')
figim(s,2.55,1.05,5.35,os.path.join(FIG,'pertf_heldout_4000.png'))
bullets(s,0.6,6.55,12.4,0.6,[
 'masked-GO+MMD is positive on 33/49 held-out TFs (ESM-kNN 34/49) with fewer negatives — a broad win, not a few lucky TFs; '
 'the old retrfix "top" model was positive on only 15/49.'],size=12.5,color=DARK)

# ═══ 9 · Epistasis singles vs combo ═══
s=slide_blank(); title_bar(s,'9 · Epistasis — singles vs combo-trained (same held-out combos)')
figim(s,0.5,1.15,4.9,os.path.join(FIG,'epistasis_singles_vs_combo.png'),w=12.3)
box(s,0.4,6.25,12.55,1.0,fill=RGBColor(0xF3,0xEC,0xFB),line=PURPLE)
text(s,0.65,6.4,12.0,0.85,'Combo-training (50/50 split) makes the do-operator BEAT additive AND predict which pairs are synergistic: '
     'epistasis r jumps −0.49 → +0.91. The mutilated-SCM machinery earns its keep only once trained on combos.',
     size=13,bold=True,color=PURPLE)
foot(s,'Same MMD backbone, same dataset (Joung), same held-out combos; do_mean = per-single-mean u composer.')

# ═══ 10 · Master table ═══
s=slide_blank(); title_bar(s,'10 · Master results (held-out, drift-removed)')
label(s,0.5,1.0,12,'Singles — n=49 held-out TFs',color=DARKBLUE,size=15)
rows=[['arm','disc','Pnk','allDEG','A-gain (ΔallDEG)','verdict'],
 ['mmd1_deg2  (winner)','0.54','0.24','0.35','+0.72','beats copy'],
 ['mmd2','0.58','0.20','0.33','+0.66','≈ copy'],
 ['mmd2_deg2','0.58','0.22','0.29','+0.66','—'],
 ['nlSCM_mmd2_deg2','0.59','0.20','0.29','+0.63','biggest A-gain'],
 ['mmd1_retr_k20','0.63','0.24','0.29','+0.64','anchor helped'],
 ['masked_go+MMD (base)','0.56','0.20','0.27','+0.62','prior best'],
 ['free357 (free decoder)','0.58','0.13','0.23','+0.56','worse; keep mask'],
 ['ESM-kNN copy (baseline)','0.60','0.21','0.31','—','the bar to beat']]
table(s,0.4,1.45,7.9,rows,[2.5,0.85,0.85,0.95,1.75,1.85],fs=11,hfs=11)
label(s,8.6,1.0,4.5,'Combo — n=11 held-out combos',color=PURPLE,size=15)
rows2=[['','singles','combo'],
 ['do_mean pearson','0.20','0.37'],
 ['do_mean allDEG','0.30','0.56'],
 ['vs model-additive','tie','+0.06'],
 ['predicts epistasis r','−0.49','+0.91']]
table(s,8.6,1.45,4.35,rows2,[2.05,1.15,1.15],fs=12,hfs=12,hcolor=PURPLE)
box(s,8.6,4.15,4.35,2.5,fill=RGBColor(0xF5,0xF5,0xEA),line=AMBER)
text(s,8.8,4.3,3.95,2.3,'Two wins:\n(1) MMD + DE-gene MSE is the first arm to beat the dumb copy on held-out DE prediction.\n'
     '(2) combo-training lifts the do-op from additive (r ≈ 0) to genuine epistasis prediction (r ≈ +0.9).',size=13,bold=True,color=DARK)
foot(s,'Singles @4000 ctrl / 3 seeds; A-gain = real − shuffled A. Combo: 50/50 split, small n → wide CI (effect large & consistent across Joung + joint arms).')

# ═══ 11 · Next steps ═══
s=slide_blank(); title_bar(s,'11 · Next steps')
box(s,0.4,1.1,12.5,1.3,fill=RGBColor(0xEC,0xF7,0xEC),line=GREEN)
text(s,0.65,1.25,12.0,1.1,'Bottom line: we broke the generalization wall (beat the copy) and cracked epistasis (combo-training). '
     'Both with the two diagnosed levers — MMD (distribution match) + drift-removed DE-gene MSE — on a load-bearing A.',size=15,bold=True,color=RGBColor(0x14,0x60,0x14))
r1=['Confirm & tighten','Widen the DE-gene MSE weight sweep around the winner (×1/3/5) + seeds; confirm the copy-beating margin with more held-out TFs.']
r2=['Scale combos','More held-out combos (small n now) + joint Joung+Norman; push epistasis r and the high-|ε| quartile.']
r3=['Generalization','Cross-dataset pretraining of the ESM→response map; the ESM-smoothness idea needs a better scale / gate-head coverage.']
r4=['Design / RL','With a combo model that predicts synergy, plan TF cocktails toward target fates over the frozen world model.']
for i,(h,b) in enumerate([r1,r2,r3,r4]):
    x=0.4+(i%2)*6.35; y=2.7+(i//2)*2.0
    box(s,x,y,6.15,1.8,fill=RGBColor(0xEC,0xF1,0xFB),line=BLUE)
    label(s,x+0.25,y+0.12,5.6,h,color=BLUE,size=15); text(s,x+0.25,y+0.6,5.7,1.1,b,size=12.5,color=DARK)
foot(s,'exp20_load_bearing · lb9 sweep · prior-off · Joung vocab-v2b.')

prs.save(OUT); print('saved', OUT, '|', len(prs.slides._sldIdLst), 'slides')
