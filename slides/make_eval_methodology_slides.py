"""
Slides explaining the evaluation methodology used in CausalBioRL:
  1. Title
  2. Observed DE — how DEGs are defined (Mann-Whitney U on Norman/Replogle cells)
  3. Single-pert evaluation — model prediction vs correlation / GRN baselines,
     AUROC / Spearman / Pearson definitions
  4. Combo evaluation — additive baseline + u-score parallel combo, and why
     additive is the floor our model has to beat.

Run: python slides/make_eval_methodology_slides.py
Output: slides/eval_methodology.pptx

Style and helpers follow `slides/make_u_score_slides.py`.
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'eval_methodology.pptx')
EQ_DIR = os.path.join(HERE, '_eq_cache')
os.makedirs(EQ_DIR, exist_ok=True)

# Palette (matches u_score deck)
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
BLACK      = RGBColor(0x00, 0x00, 0x00)
DARK       = RGBColor(0x1A, 0x1A, 0x2E)
BLUE       = RGBColor(0x21, 0x71, 0xB5)
GRAY       = RGBColor(0x66, 0x66, 0x66)
LIGHT_GRAY = RGBColor(0xF5, 0xF5, 0xF5)
RED        = RGBColor(0xC0, 0x39, 0x2B)
GREEN      = RGBColor(0x27, 0xAE, 0x60)
AMBER      = RGBColor(0xD9, 0x82, 0x1E)
DARKBLUE   = RGBColor(0x0B, 0x3D, 0x91)

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)


def add_blank_slide():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background; bg.fill.solid(); bg.fill.fore_color.rgb = WHITE
    return slide


def add_text(slide, left, top, width, height, text, *,
             size=18, bold=False, color=BLACK, align=PP_ALIGN.LEFT,
             font='Calibri', italic=False):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = text
    p.font.size = Pt(size); p.font.bold = bold; p.font.italic = italic
    p.font.color.rgb = color; p.font.name = font
    p.alignment = align
    return tb


def add_bullets(slide, left, top, width, height, items, *,
                size=15, color=BLACK, spacing=4, bold_first_word=False):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame; tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if bold_first_word and '. ' in item:
            head, rest = item.split('. ', 1)
            r1 = p.add_run(); r1.text = head + '. '
            r1.font.size = Pt(size); r1.font.bold = True; r1.font.color.rgb = DARK
            r1.font.name = 'Calibri'
            r2 = p.add_run(); r2.text = rest
            r2.font.size = Pt(size); r2.font.color.rgb = color
            r2.font.name = 'Calibri'
        else:
            p.text = '• ' + item
            p.font.size = Pt(size); p.font.color.rgb = color
            p.font.name = 'Calibri'
        p.space_after = Pt(spacing)
    return tb


def add_title_bar(slide, title_text, *, color=DARK):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.85))
    shape.fill.solid(); shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    add_text(slide, 0.5, 0.15, 12.3, 0.6, title_text,
             size=26, bold=True, color=WHITE)


def add_section_label(slide, left, top, width, text, *, color=DARKBLUE, size=18):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(0.4))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = text
    p.font.size = Pt(size); p.font.bold = True; p.font.color.rgb = color
    p.font.name = 'Calibri'
    return tb


def add_box(slide, left, top, width, height, *, fill=LIGHT_GRAY, line=None):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid(); shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(1.5)
    shape.shadow.inherit = False
    return shape


def render_eq(name, tex, *, fontsize=22, color='#0B3D91', dpi=300, pad=0.18):
    """Render a single LaTeX expression to a PNG (mathtext)."""
    path = os.path.join(EQ_DIR, f'{name}.png')
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.patch.set_alpha(0)
    txt = fig.text(0.0, 0.0, tex, fontsize=fontsize, color=color,
                   ha='left', va='bottom')
    fig.canvas.draw()
    bbox = txt.get_window_extent()
    w, h = bbox.width / fig.dpi, bbox.height / fig.dpi
    fig.set_size_inches(w + 2 * pad, h + 2 * pad)
    txt.set_position((pad / (w + 2 * pad), pad / (h + 2 * pad)))
    fig.savefig(path, dpi=dpi, transparent=True, bbox_inches='tight', pad_inches=pad)
    plt.close(fig)
    return path


def add_eq(slide, left, top, name, tex, *, width=None, fontsize=22,
           color='#0B3D91', dpi=300):
    path = render_eq(name, tex, fontsize=fontsize, color=color, dpi=dpi)
    if width is not None:
        slide.shapes.add_picture(path, Inches(left), Inches(top), width=Inches(width))
    else:
        slide.shapes.add_picture(path, Inches(left), Inches(top))


def render_eq_block(name, lines, *, fontsize=20, color='#0B3D91', dpi=300,
                    line_spacing=1.45, pad=0.25):
    path = os.path.join(EQ_DIR, f'{name}.png')
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    n = len(lines)
    texts = []
    for i, ln in enumerate(lines):
        y = 1.0 - (i + 0.5) / n
        t = ax.text(0.5, y, ln, fontsize=fontsize, color=color,
                    ha='center', va='center')
        texts.append(t)
    fig.canvas.draw()
    bboxes = [t.get_window_extent() for t in texts]
    max_w = max(b.width for b in bboxes) / fig.dpi
    line_h = (fontsize / 72.0) * line_spacing
    h = n * line_h
    w = max_w
    fig.set_size_inches(w + 2 * pad, h + 2 * pad)
    fig.savefig(path, dpi=dpi, transparent=True, bbox_inches='tight', pad_inches=pad)
    plt.close(fig)
    return path


def add_eq_block(slide, left, top, name, lines, *, width=None, fontsize=20,
                 color='#0B3D91', dpi=300, line_spacing=1.45):
    path = render_eq_block(name, lines, fontsize=fontsize, color=color, dpi=dpi,
                           line_spacing=line_spacing)
    if width is not None:
        slide.shapes.add_picture(path, Inches(left), Inches(top), width=Inches(width))
    else:
        slide.shapes.add_picture(path, Inches(left), Inches(top))


# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ═══════════════════════════════════════════════════════════════════════════
slide = add_blank_slide()
add_text(slide, 1, 1.9, 11, 1.3,
         'Perturbation Eval — Methodology',
         size=40, bold=True, color=DARK, align=PP_ALIGN.CENTER)
add_text(slide, 1, 3.2, 11, 1.2,
         'How DEGs are called, how the correlation and GRN baselines work,\n'
         'and exactly what the additive combo baseline computes.',
         size=18, color=GRAY, align=PP_ALIGN.CENTER)
add_text(slide, 1, 5.6, 11, 0.4,
         'CausalBioRL  |  evaluation/run_perturbation_eval.py + eval_combinatorial.py',
         size=14, color=GRAY, align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Observed DEGs: step-by-step with real cell counts
# ═══════════════════════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, 'Step 1 — Build the DEG ground truth (one example: DONSON in Replogle)')

# Left column: step-by-step recipe
add_box(slide, 0.3, 1.05, 6.55, 5.1, fill=LIGHT_GRAY)
add_section_label(slide, 0.5, 1.15, 6.2,
                  'Per perturbation g — what we do:',
                  color=DARKBLUE, size=17)

add_bullets(slide, 0.5, 1.65, 6.2, 4.4, [
    '1.  Find pert cells: indices where obs.gene == g.  '
    'For DONSON: 688 cells.',
    '2.  Compare to ALL control cells (obs.gene == "non-targeting"): '
    '10,691 cells in Replogle K562, 11,183 in Norman.',
    '3.  Per gene g′ in the 2,000-HVG axis, compute lfc:  '
    'mean(pert) − mean(ctrl).  Signed.',
    '4.  Per gene g′, run Mann–Whitney U (rank-sum test on pert vs ctrl '
    'distributions) to get a p-value.  Vectorised across all 2,000 genes.',
    '5.  A gene is a DEG of pert g iff its p-value < 0.05.  '
    'For DONSON: 892 of 2,000 HVGs flagged.',
    '6.  Pooled effect size = |lfc| / sqrt((σ²_ctrl + σ²_pert) / 2).  '
    'Used for ranking metrics that need a non-negative magnitude.',
    '7.  Perturbations with fewer than MIN_CELLS = 5 pert cells are dropped.',
], size=12, color=BLACK, spacing=5)

# Right column: cohort numbers / filtering — the "real numbers"
add_box(slide, 7.0, 1.05, 6.0, 5.1, fill=LIGHT_GRAY, line=BLUE)
add_section_label(slide, 7.2, 1.15, 5.7,
                  'Cohort sizes after filtering',
                  color=BLUE, size=17)

# Replogle block
add_text(slide, 7.2, 1.65, 5.6, 0.35,
         'Replogle K562 (train_v5.h5ad, CRISPRi)',
         size=13, bold=True, color=DARKBLUE)
add_bullets(slide, 7.2, 1.95, 5.6, 1.7, [
    '186,807 cells after cell_line == "k562" filter',
    '10,691 non-targeting control cells',
    '1,371 perturbed genes (≥ 5 cells each)',
    'Gene split:  1,097 train / 137 val / 137 test  (seed 42)',
    'On test split: 40 of 137 have ≥ 3 DEGs (AUROC computable)',
], size=11, color=BLACK, spacing=2)

# Norman block
add_text(slide, 7.2, 3.85, 5.6, 0.35,
         'Norman (MTX, K562 CRISPRa)',
         size=13, bold=True, color=DARKBLUE)
add_bullets(slide, 7.2, 4.15, 5.6, 1.95, [
    '111,668 raw barcodes → 104,507 after good_coverage filter',
    '11,183 non-targeting control cells',
    '105 single-pert genes, 131 combos',
    'On Replogle 2,000-HVG axis: 19 / 105 singles, 4 / 131 combos are '
    'evaluable (the rest have target gene outside the HVG axis)',
    'K562 essential unfiltered (8,563 genes) would lift coverage to ~54 / 105',
], size=11, color=BLACK, spacing=2)

# Bottom verdict bar
add_box(slide, 0.3, 6.3, 12.7, 0.85, fill=RGBColor(0xEC, 0xF7, 0xEC), line=GREEN)
add_text(slide, 0.55, 6.45, 12.3, 0.5,
         'The DEG mask is the SAME ground truth for the model, correlation '
         'baseline, GRN baseline, and additive baseline.',
         size=14, color=DARK)


# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Single-pert evaluation (model + 2 baselines + 4 metrics)
# ═══════════════════════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, 'Step 2 — Single-pert: Model vs. Correlation vs. GRN Baselines')

# Row 1 — three boxes for the three predictors
PRED_W = 4.15
PRED_H = 2.4
PRED_TOP = 1.05

# 1) Causal VAE model
add_box(slide, 0.3, PRED_TOP, PRED_W, PRED_H, fill=LIGHT_GRAY, line=BLUE)
add_section_label(slide, 0.45, PRED_TOP + 0.08, PRED_W - 0.3,
                  'Causal VAE (D†)', color=BLUE, size=15)
add_eq_block(slide, 0.4, PRED_TOP + 0.55, 'pred_model', [
    r'$\hat z \;=\; \mathrm{intervene\_gene\_u}(\varepsilon,\,\Delta x_g,\,A,\,g)$',
    r'$\widehat{\Delta x}^{\,\mathrm{model}}_g \;=\; \mathrm{mean}(D\hat z) - \mathrm{mean}(D z^{\,\mathrm{ctrl}})$',
], width=PRED_W - 0.15, fontsize=14, line_spacing=1.55)
add_text(slide, 0.45, PRED_TOP + 1.65, PRED_W - 0.3, 0.7,
         'Sample 50 random ctrl cells (seed=42), encode → ε, draw ONE A '
         'sample (n_dag_samples=1 deterministic at eval), apply u-score do(g), '
         'decode, average the 50 cell predictions.',
         size=10, color=BLACK)

# 2) Correlation baseline
add_box(slide, 4.6, PRED_TOP, PRED_W, PRED_H, fill=LIGHT_GRAY, line=GRAY)
add_section_label(slide, 4.75, PRED_TOP + 0.08, PRED_W - 0.3,
                  'Correlation baseline', color=GRAY, size=15)
add_eq_block(slide, 4.7, PRED_TOP + 0.55, 'pred_corr', [
    r'$C_{ij} \;=\; |\,\mathrm{Pearson}(X^{\,\mathrm{ctrl}}_{:,i},\,X^{\,\mathrm{ctrl}}_{:,j})\,|$',
    r'$\mathrm{score}^{\,\mathrm{corr}}_g \;=\; C_{g,\,:}$',
], width=PRED_W - 0.15, fontsize=14, line_spacing=1.55)
add_text(slide, 4.75, PRED_TOP + 1.65, PRED_W - 0.3, 0.7,
         'C built from ALL ctrl cells (Replogle 10,691 or Norman 11,183). '
         'Score row g excluded self. No causal claim — purely co-variation.',
         size=10, color=BLACK)

# 3) GRN baseline
add_box(slide, 8.9, PRED_TOP, PRED_W, PRED_H, fill=LIGHT_GRAY,
        line=RGBColor(0xFD, 0x8D, 0x3C))
add_section_label(slide, 9.05, PRED_TOP + 0.08, PRED_W - 0.3,
                  'Learned GRN  (D · A · Dᵀ)', color=RGBColor(0xFD, 0x8D, 0x3C), size=15)
add_eq_block(slide, 9.0, PRED_TOP + 0.55, 'pred_grn', [
    r'$G \;=\; D\,\bar A\,D^{\top}$',
    r'$\mathrm{score}^{\,\mathrm{grn}}_g \;=\; |\,G_{g,\,:}\,|$',
], width=PRED_W - 0.15, fontsize=14, line_spacing=1.55)
add_text(slide, 9.05, PRED_TOP + 1.65, PRED_W - 0.3, 0.7,
         'Uses the model parameters directly: D and the mean edge prob A from '
         'dag.get_edge_probs(). One static G — no per-cell sampling.',
         size=10, color=BLACK)

# Row 2 — the 4 metrics
METRIC_TOP = 3.7
add_section_label(slide, 0.4, METRIC_TOP, 13, 'Metrics computed per perturbation, then averaged:',
                  color=DARK, size=16)

# Layout 4 metric boxes
m_h = 2.35
m_w = 3.1
m_y = METRIC_TOP + 0.45
gap = 0.18

# AUROC
x0 = 0.4
add_box(slide, x0, m_y, m_w, m_h, fill=RGBColor(0xF0, 0xF0, 0xF0))
add_text(slide, x0 + 0.15, m_y + 0.08, m_w - 0.3, 0.35,
         'AUROC (DEG detection)', size=14, bold=True, color=DARK)
add_eq_block(slide, x0 + 0.08, m_y + 0.55, 'm_auroc', [
    r'$\mathrm{AUROC}\!\left(\,|\widehat{\Delta x}|,\;[\,p_g < 0.05\,]\,\right)$',
], width=m_w - 0.1, fontsize=13, line_spacing=1.5)
add_text(slide, x0 + 0.15, m_y + 1.15, m_w - 0.3, 1.1,
         'Can the score rank true DEGs above non-DEGs?  0.5 = random.',
         size=10, color=BLACK)

# Spearman
x0 += m_w + gap
add_box(slide, x0, m_y, m_w, m_h, fill=RGBColor(0xF0, 0xF0, 0xF0))
add_text(slide, x0 + 0.15, m_y + 0.08, m_w - 0.3, 0.35,
         'Spearman ρ on |effect|', size=14, bold=True, color=DARK)
add_eq_block(slide, x0 + 0.08, m_y + 0.55, 'm_spear', [
    r'$\rho_S\!\left(\,|\widehat{\Delta x}|,\; \mathrm{eff}\,\right)$',
], width=m_w - 0.1, fontsize=13, line_spacing=1.5)
add_text(slide, x0 + 0.15, m_y + 1.15, m_w - 0.3, 1.1,
         'How well the score RANKS genes by effect-size magnitude. '
         'Signed variant uses lfc instead.',
         size=10, color=BLACK)

# Pearson on DEGs
x0 += m_w + gap
add_box(slide, x0, m_y, m_w, m_h, fill=RGBColor(0xF0, 0xF0, 0xF0))
add_text(slide, x0 + 0.15, m_y + 0.08, m_w - 0.3, 0.35,
         'Pearson on DEGs', size=14, bold=True, color=DARK)
add_eq_block(slide, x0 + 0.08, m_y + 0.55, 'm_pear', [
    r'$r\!\left(\,\widehat{\Delta x}_{S},\; \Delta x_{S}\,\right),\;\; S = \{g : |\Delta x_g| > 0.05\}$',
], width=m_w - 0.1, fontsize=12, line_spacing=1.5)
add_text(slide, x0 + 0.15, m_y + 1.15, m_w - 0.3, 1.1,
         'Linear correlation of predicted vs observed lfc, computed only on '
         'genes with |Δx| > 0.05 (DEGs).',
         size=10, color=BLACK)

# Direction accuracy
x0 += m_w + gap
add_box(slide, x0, m_y, m_w, m_h, fill=RGBColor(0xF0, 0xF0, 0xF0))
add_text(slide, x0 + 0.15, m_y + 0.08, m_w - 0.3, 0.35,
         'Direction acc. (DEGs)', size=14, bold=True, color=DARK)
add_eq_block(slide, x0 + 0.08, m_y + 0.55, 'm_dir', [
    r'$\frac{1}{|S|}\,\sum_{g \in S}\,\left[\,\mathrm{sgn}(\widehat{\Delta x}_g) = \mathrm{sgn}(\Delta x_g)\,\right]$',
], width=m_w - 0.1, fontsize=12, line_spacing=1.5)
add_text(slide, x0 + 0.15, m_y + 1.5, m_w - 0.3, 0.8,
         'Fraction of DEGs with correct sign (up vs down). S = sig set.',
         size=10, color=BLACK)


# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Combo: additive baseline & parallel u-score combo
# ═══════════════════════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, 'Step 3 — Combo (Norman only): Additive Baseline & Parallel u-score Combo')

# Left: additive baseline
add_box(slide, 0.3, 1.05, 6.3, 4.0, fill=LIGHT_GRAY, line=GREEN)
add_section_label(slide, 0.5, 1.15, 6.0, 'Additive baseline', color=GREEN, size=18)

add_eq_block(slide, 0.4, 1.6, 'combo_add', [
    r'$\widehat{\Delta x}^{\,\mathrm{combo}}_{\mathrm{additive}}(g_a, g_b) \;=\; \Delta x_{g_a} \;+\; \Delta x_{g_b}$',
], width=6.1, fontsize=20)

add_text(slide, 0.5, 2.55, 6.0, 0.4,
         'where each Δx_{g} is the OBSERVED single-pert delta from Norman cells.',
         size=13, italic=True, color=DARK)

add_bullets(slide, 0.5, 2.95, 6.0, 2.1, [
    'Δx_{g_a}, Δx_{g_b} come from Norman OBSERVED single-pert lfc '
    '(de_single dict, full single-pert cells, not held out).',
    'No epistasis, no saturation — pure linear superposition.',
    'Strong because ~70% of combo effects are near-additive (Norman 2019).',
    'Combo cells held out for eval: 20 ctrl cells (seed=42) used for the model '
    'pipeline; additive uses ALL ctrl_mean.',
], size=11, color=BLACK, spacing=4)

# Right: parallel u-score combo
add_box(slide, 6.75, 1.05, 6.25, 4.0, fill=LIGHT_GRAY, line=BLUE)
add_section_label(slide, 6.95, 1.15, 6.0, 'Causal VAE (parallel u-score)', color=BLUE, size=18)

add_eq_block(slide, 6.85, 1.6, 'combo_uscore', [
    r'$u_{\mathrm{combo}} \;=\; \mathrm{clip}\!\left(\,L_{g_a,\,:} + L_{g_b,\,:},\;0,\;u_{\max}\right)$',
    r'$\Delta z_{\mathrm{combo}} \;=\; D^{+}\,(\Delta x_{g_a} + \Delta x_{g_b})$',
    r'$\tilde\varepsilon \;=\; u_{\mathrm{combo}} \odot (z^{\,\mathrm{prev}} + \Delta z_{\mathrm{combo}}) + (1 - u_{\mathrm{combo}}) \odot \varepsilon$',
    r'$\hat z \;=\; (I - M\,A^{\top})^{-1}\,\tilde\varepsilon$',
], width=6.05, fontsize=14, line_spacing=1.55)

add_text(slide, 6.9, 4.45, 6.0, 0.4,
         'Order-invariant (AB ≡ BA by construction).  M = diag(1−u_combo).',
         size=12, italic=True, color=DARK)

# Bottom verdict bar
add_box(slide, 0.3, 5.25, 12.7, 1.85, fill=RGBColor(0xFF, 0xF4, 0xE5), line=AMBER)
add_text(slide, 0.55, 5.4, 12.3, 0.4,
         'Why the additive baseline is hard to beat:',
         size=15, bold=True, color=AMBER)
add_eq_block(slide, 0.45, 5.85, 'combo_collapse', [
    r'$A \to 0 \;\;\Longrightarrow\;\; (I-M\,A^{\top})^{-1} \to I \;\;\Longrightarrow\;\; \hat z \to \tilde\varepsilon$',
    r'$\;\;\Longrightarrow\;\; \widehat{\Delta x}^{\,\mathrm{combo}}_{\mathrm{model}} \;\approx\; D\,D^{+}\,(\Delta x_{g_a} + \Delta x_{g_b}) \;\approx\; \widehat{\Delta x}^{\,\mathrm{combo}}_{\mathrm{additive}}$',
], width=12.4, fontsize=15, line_spacing=1.55)


# ═══════════════════════════════════════════════════════════════════════════
prs.save(OUT)
print(f'Wrote {OUT}')
