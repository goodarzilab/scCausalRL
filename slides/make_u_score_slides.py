"""
Generate slides for the u-score (decoder-loading) intervention scheme:
diagnostic, problems with current model, and mathematical derivation of the
new do-operator (with LaTeX-rendered math via matplotlib mathtext).

Run: python slides/make_u_score_slides.py
Output: slides/u_score_architecture.pptx
"""
import os
import io
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'u_score_architecture.pptx')
EQ_DIR = os.path.join(HERE, '_eq_cache')
os.makedirs(EQ_DIR, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────
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


# ─── helpers ──────────────────────────────────────────────────────────────
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


# ─── math rendering via matplotlib mathtext ────────────────────────────────
def render_eq(name, tex, *, fontsize=22, color='#0B3D91', dpi=300, pad=0.18,
              align='center'):
    """Render a single LaTeX expression to a PNG (via matplotlib mathtext)
    and return the file path. Cached by `name`.
    """
    path = os.path.join(EQ_DIR, f'{name}.png')
    # Always re-render so style stays consistent if we tweak.
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
    """Render an equation and place it at (left, top) inches. If `width` is
    given, the picture is scaled to that width."""
    path = render_eq(name, tex, fontsize=fontsize, color=color, dpi=dpi)
    if width is not None:
        slide.shapes.add_picture(path, Inches(left), Inches(top), width=Inches(width))
    else:
        slide.shapes.add_picture(path, Inches(left), Inches(top))


def render_eq_block(name, lines, *, fontsize=20, color='#0B3D91', dpi=300,
                    line_spacing=1.45, pad=0.25):
    """Render a *block* of equations (list of TeX lines) as one PNG."""
    path = os.path.join(EQ_DIR, f'{name}.png')
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    # Render each line and track widths
    n = len(lines)
    # Use axes coordinates: y top-down
    texts = []
    for i, ln in enumerate(lines):
        y = 1.0 - (i + 0.5) / n
        t = ax.text(0.5, y, ln, fontsize=fontsize, color=color,
                    ha='center', va='center')
        texts.append(t)
    fig.canvas.draw()
    # Measure overall bbox
    bboxes = [t.get_window_extent() for t in texts]
    max_w = max(b.width for b in bboxes) / fig.dpi
    # height: roughly fontsize * line_spacing per line in inches
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
         'DAG Diagnosis & the u-score Do-Operator',
         size=40, bold=True, color=DARK, align=PP_ALIGN.CENTER)
add_text(slide, 1, 3.2, 11, 0.7,
         'Why the DAG collapsed to its initialization, why the threshold-mutilation\n'
         'pipeline is biased, and a principled replacement.',
         size=18, color=GRAY, align=PP_ALIGN.CENTER)
add_text(slide, 1, 5.6, 11, 0.4,
         'scCausalRL  |  Phase-1 deep-dive  |  Goodarzi Lab, Arc Institute',
         size=14, color=GRAY, align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 2 — The diagnostic
# ═══════════════════════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, 'Phase-1 Diagnostic — checkpoints/phase1_best_no_dag.pt')

# Left column: DAG structure
add_box(slide, 0.4, 1.2, 6.2, 2.55, fill=LIGHT_GRAY)
add_section_label(slide, 0.6, 1.3, 6.0, 'DAG structure', color=DARKBLUE, size=18)
add_bullets(slide, 0.6, 1.75, 6.0, 2.0, [
    'mean edge prob:     0.0071',
    'median edge prob:   0.0064',
    'max edge prob:      0.0465',
    'θ mean / std:       −5.056  /  0.491',
    'edges (p > 0.1):    0  /  4032',
    'edges (p > 0.3, 0.5, 0.7):  0',
], size=14, color=BLACK, spacing=2)

# Left column: orthogonality
add_box(slide, 0.4, 3.85, 6.2, 1.75, fill=LIGHT_GRAY)
add_section_label(slide, 0.6, 3.95, 6.0, 'Orthogonality (decoder D)', color=DARKBLUE, size=18)
add_bullets(slide, 0.6, 4.4, 6.0, 1.3, [
    'off-diag mean(|·|): 0.0006',
    'off-diag max(|·|):  0.0036',
    'col-norm CV:        0.0007   (highly uniform)',
], size=14, color=BLACK, spacing=2)

# Right column: mutilation
add_box(slide, 6.85, 1.2, 6.2, 4.4, fill=LIGHT_GRAY)
add_section_label(slide, 7.05, 1.3, 6.0,
                  'Mutilation (threshold scheme, 137 test perts, N=8 DAGs)',
                  color=DARKBLUE, size=18)
add_bullets(slide, 7.05, 1.95, 6.0, 4.0, [
    'mean n_affected:    48.6 ± 2.3   (max d = 64)',
    'median n_affected:  49',
    'frac > 10 affected: 100%',
    'frac > 30 affected: 100%',
    'frac ≤  5 affected:   0%',
    'mean max |dε|:      0.292',
    'dε  L1/L2 :         6.25     (1=sparse, √64=8=dense)',
    'dz  L1/L2 :         6.17',
], size=14, color=BLACK, spacing=2)

# Bottom verdict bar
add_box(slide, 0.4, 5.85, 12.65, 1.3, fill=RGBColor(0xFF, 0xF4, 0xE5), line=AMBER)
add_text(slide, 0.6, 5.95, 12.3, 0.4,
         'Verdict (printed by dag_diagnostics.py):',
         size=14, bold=True, color=AMBER)
add_text(slide, 0.6, 6.35, 12.3, 0.4,
         '⚠  Most programs mutilated per pert (49/64) — dε is dense, not sparse',
         size=15, color=BLACK)
add_text(slide, 0.6, 6.75, 12.3, 0.4,
         '✓  D is nearly orthogonal (off-diag mean = 0.001)',
         size=15, color=GREEN)


# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Two distinct problems
# ═══════════════════════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, 'Two Distinct Problems Behind That Output')

# Problem A: DAG never left init
add_box(slide, 0.4, 1.1, 6.25, 5.3, fill=LIGHT_GRAY, line=RED)
add_section_label(slide, 0.6, 1.25, 6.0,
                  'Problem A  —  the DAG never left initialization',
                  color=RED, size=18)
add_bullets(slide, 0.6, 1.8, 6.0, 2.6, [
    'θ ~ N(−5.0, 0.5²)  →  edge prob ≈ σ(−5) ≈ 0.0067',
    'Observed mean: 0.0071  →  literally still the init.',
    'λ_sparse = 1.0 actively pulls θ down.',
    'λ_prior = 0.1, margin = 0.1 → margin-ranking too weak.',
    'no_dag checkpoint sets ρ_dagma = 0 → no force pushing up.',
], size=14, color=BLACK)

add_text(slide, 0.6, 4.5, 6.0, 0.4,
         'Smoking gun:', size=15, bold=True, color=RED)
add_eq(slide, 1.6, 4.95, 'eq_sigmoid',
       r'$\sigma(-5)\;=\;0.00669\;\;\approx\;\;0.0071\;\;(\mathrm{observed})$',
       fontsize=20, color='#C0392B', width=4.5)
add_text(slide, 0.6, 5.55, 6.0, 0.4,
         'Fix:  grid-search (λ_sparse, λ_prior, margin) on a 5k-cell Replogle subset.',
         size=13, color=BLACK, italic=True)

# Problem B: threshold scheme is itself dense
add_box(slide, 6.85, 1.1, 6.2, 5.3, fill=LIGHT_GRAY, line=AMBER)
add_section_label(slide, 7.05, 1.25, 6.0,
                  'Problem B  —  threshold scheme is dense by construction',
                  color=AMBER, size=18)
add_bullets(slide, 7.05, 1.8, 6.0, 2.6, [
    'Pipeline: dz = D⁺·dx,  dε = (I − A^T) dz,  mutilate where |dε|>τ·max.',
    'When A ≈ 0 (Problem A!),  dε ≈ dz.',
    'dz = D⁺ dx for a single gene is dense — orthogonality controls D columns, not rows.',
    'So threshold catches half the latent dim regardless of biology.',
    'Fix #1: u-score (slides 5–8).  Fix #2: add L1 on D entries (gene-level sparsity).',
], size=13, color=BLACK)

add_text(slide, 7.05, 4.5, 6.0, 0.4,
         'Math says:', size=15, bold=True, color=AMBER)
add_eq(slide, 7.55, 4.95, 'eq_threshold_bias',
       r'$z^{\mathrm{thr}}_{t,i}\;=\;z_{t-1,i}+dz_i\;-\;(A^{\!\top}(z_{t-1}+dz))_i$',
       fontsize=18, color='#D9821E', width=5.5)
add_text(slide, 7.05, 5.55, 6.0, 0.4,
         'Residual only vanishes when A→0.  Fix: u-score do-operator (slides 5–8).',
         size=13, color=BLACK, italic=True)

# Footer
add_text(slide, 0.5, 6.7, 12.3, 0.4,
         'These problems compound: Problem A hides Problem B at training time, '
         'and Problem B prevents the gradient signal that would fix Problem A.',
         size=13, color=GRAY, italic=True, align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 4 — The linear SCM (setup for the math)
# ═══════════════════════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, 'Setup — the Linear Latent SCM and the Decoder')

add_section_label(slide, 0.5, 1.1, 12, 'Causal VAE generative model')
add_eq_block(slide, 1.4, 1.55, 'eq_setup',
             [r'$z \;=\; A^{\!\top} z \;+\; \varepsilon \;\;\;\Longleftrightarrow\;\;\; '
              r'z \;=\; (I - A^{\!\top})^{-1}\,\varepsilon$',
              r'$x \;=\; D\, z \;+\; b$'],
             fontsize=24, color='#0B3D91', dpi=300)

add_section_label(slide, 0.5, 3.5, 12, 'What we want to intervene on')
add_bullets(slide, 0.7, 3.95, 12, 2.0, [
    'A gene-level action delivers dx ∈ R^G  (supported on one row in the gene axis).',
    'World model must lift dx → a latent intervention on z under the SCM.',
    'Two candidate liftings: (i) argmax|D[g,:]|  (ii) D⁺ + threshold mutilation.',
    '(i) ignores the long tail; (ii) is biased when A is small (Theorem 3, slide 7).',
], size=15, color=BLACK)

add_section_label(slide, 0.5, 6.0, 12, 'Notation we will keep using',
                  color=GRAY, size=15)
add_eq(slide, 0.5, 6.4, 'eq_notation',
       r'$d\!=\!64\;\;\;\;G\!=\!2000\;\;\;\;\varepsilon\!\in\!\mathbb{R}^{d}'
       r'\;\;\;\;A\!\in\![0,1]^{d\times d}\;(\mathrm{diag}\!=\!0)'
       r'\;\;\;\;D\!\in\!\mathbb{R}^{G\times d}\;\mathrm{near\!-\!orthogonal}$',
       fontsize=18, color='#666666', width=12.3)


# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 5 — The u-score scheme (definition)
# ═══════════════════════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, 'The u-score Do-Operator — Definition')

# Loading matrix
add_section_label(slide, 0.5, 1.0, 6, 'Decoder loadings L')
add_eq_block(slide, 0.7, 1.4, 'eq_L',
             [r'$T_i \;=\; \sum_{h=1}^{G} D_{hi}^{2}$',
              r'$L_{g,i} \;=\; \dfrac{D_{g,i}^{2}}{T_i}\;\in\;[0,1]$'],
             fontsize=20, color='#0B3D91', dpi=300)
add_bullets(slide, 0.7, 2.95, 6.0, 1.4, [
    'L_{g,i} = fraction of program-i decoder energy from gene g.',
    'Σ_g L_{g,i} = 1   (each column of L is a distribution over genes).',
    'L ≥ 0, computed once per training epoch (cheap).',
], size=13, color=BLACK)

# Strength vector accumulation
add_section_label(slide, 6.85, 1.0, 6.2, 'Strength vector u (per episode)')
add_eq_block(slide, 7.0, 1.4, 'eq_u_update',
             [r'$u_{0}\;=\;0$',
              r'$u_{t} \;=\; \mathrm{clip}\!(u_{t-1} + L_{g_t,:},\;0,\;u_{\max})$'],
             fontsize=20, color='#0B3D91', dpi=300)
add_bullets(slide, 7.05, 2.95, 6.0, 1.4, [
    'u_t monotone non-decreasing → persistence is automatic.',
    'Sparse for single-gene actions (only programs g loads on get mass).',
    'No threshold to tune.',
], size=13, color=BLACK)

# Transition step
add_box(slide, 0.4, 4.4, 12.65, 2.9, fill=LIGHT_GRAY)
add_section_label(slide, 0.6, 4.5, 12, 'Transition step  (knockout / overexpress action)')
add_eq_block(slide, 0.7, 4.9, 'eq_transition',
             [r'$dz \;=\; (D^{\!\top}\!D + \lambda I)^{-1} D^{\!\top}\, dx,\qquad'
              r' z^{\mathrm{target}}_t \;=\; z_{t-1}+dz$',
              r'$\tilde\varepsilon_t \;=\; u_t \odot z^{\mathrm{target}}_t '
              r'\;+\; (1-u_t)\odot \varepsilon_{t-1}$',
              r'$M_t \;=\; \mathrm{diag}(\mathbf{1}-u_t),\qquad '
              r'\tilde{A}_t \;=\; A\,M_t,\qquad '
              r'(I - M_t A^{\!\top})\,z_t \;=\; \tilde\varepsilon_t$'],
             fontsize=20, color='#0B3D91', dpi=300, line_spacing=1.45)


# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 6 — Lemma 1 & Lemma 2
# ═══════════════════════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, 'Endpoint Lemmas — u = 0 is identity, u = e_i is classical do')

# Lemma 1
add_box(slide, 0.4, 1.05, 6.25, 5.65, fill=LIGHT_GRAY, line=BLUE)
add_section_label(slide, 0.6, 1.15, 6.0,
                  'Lemma 1 (u = 0  ⇒  identity)', color=DARKBLUE)
add_text(slide, 0.6, 1.65, 6.0, 0.35,
         'Claim:', size=13, bold=True, color=DARK)
add_eq(slide, 0.8, 1.95, 'eq_l1_claim',
       r'$u=0 \;\Rightarrow\; z_t = z_{t-1},\;\;\tilde A_t = A.$',
       fontsize=18, color='#0B3D91', width=5.5)

add_text(slide, 0.6, 2.7, 6.0, 0.35,
         'Proof:', size=13, bold=True, color=DARK)
add_eq_block(slide, 0.6, 3.0, 'eq_l1_proof',
             [r'$M_t = I \;\;\Rightarrow\;\; \tilde A_t = A\,I = A$',
              r'$\tilde\varepsilon_t = 0\!\odot\! z^{\mathrm{target}} '
              r'+ 1\!\odot\!\varepsilon_{t-1} = \varepsilon_{t-1}$',
              r'$(I-A^{\!\top})\,z_t = \varepsilon_{t-1}$',
              r'$\;\;\Rightarrow\;\; z_t = (I-A^{\!\top})^{-1}\varepsilon_{t-1} = z_{t-1}.\;\;\blacksquare$'],
             fontsize=17, color='#0B3D91', dpi=300, line_spacing=1.45)
add_text(slide, 0.6, 6.2, 6.0, 0.35,
         'Verified to numerical zero in test_u_zero_identity.',
         size=12, italic=True, color=GREEN)

# Lemma 2
add_box(slide, 6.85, 1.05, 6.2, 5.65, fill=LIGHT_GRAY, line=BLUE)
add_section_label(slide, 7.05, 1.15, 6.0,
                  'Lemma 2 (u_i = 1  ⇒  classical do)', color=DARKBLUE)
add_text(slide, 7.05, 1.65, 6.0, 0.35,
         'Claim:', size=13, bold=True, color=DARK)
add_eq(slide, 7.25, 1.95, 'eq_l2_claim',
       r'$u = e_i \;\Rightarrow\; z_{t,i} = z^{\mathrm{target}}_{t,i},\;\;\tilde A_t\, e_i = 0.$',
       fontsize=17, color='#0B3D91', width=5.5)

add_text(slide, 7.05, 2.7, 6.0, 0.35,
         'Proof:', size=13, bold=True, color=DARK)
add_eq_block(slide, 7.05, 3.0, 'eq_l2_proof',
             [r'$\tilde A_t = A(I-e_i e_i^{\!\top}) \;\Rightarrow\; \tilde A_t e_i = 0$',
              r'$\mathrm{row}\,i\;\mathrm{of}\;M_t A^{\!\top} = 0 \;\Rightarrow\; '
              r'\mathrm{row}\,i\;\mathrm{of}\;(I-M_t A^{\!\top})=e_i^{\!\top}$',
              r'$z_{t,i} = \tilde\varepsilon_{t,i} = z^{\mathrm{target}}_{t,i}.\;\;\blacksquare$'],
             fontsize=17, color='#0B3D91', dpi=300, line_spacing=1.45)
add_text(slide, 7.05, 6.2, 6.0, 0.35,
         'Verified to numerical zero in test_u_one_pins_target.',
         size=12, italic=True, color=GREEN)


# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 7 — Theorem 3 (threshold scheme is biased)
# ═══════════════════════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, 'Theorem 3 — the Threshold Scheme is Biased Away from do',
              color=RGBColor(0x80, 0x20, 0x20))

add_section_label(slide, 0.5, 1.0, 12, 'Claim')
add_eq(slide, 0.8, 1.35, 'eq_thm_claim',
       r'$\mathrm{For\;every\;}i\in\mathcal{S}\!:\;\;\;'
       r'z^{\mathrm{thr}}_{t,i} \;=\; z_{t-1,i} + dz_i '
       r'\;-\; (A^{\!\top}(z_{t-1}+dz))_i\!,'
       r'\;\;\;\mathrm{not}\;\; z_{t-1,i}+dz_i.$',
       fontsize=17, color='#0B3D91', width=12.0)

add_section_label(slide, 0.5, 2.4, 12, 'Proof')
add_eq_block(slide, 0.6, 2.75, 'eq_thm_proof',
             [r'$\tilde\varepsilon_t \;=\; \varepsilon_{t-1} + d\varepsilon, '
              r'\;\;d\varepsilon=(I-A^{\!\top})dz \;\;\;'
              r'\Rightarrow\;\;\;d\varepsilon_i \;=\; dz_i - (A^{\!\top}dz)_i$',
              r'$i\!\in\!\mathcal{S}\;\Rightarrow\;\mathrm{row}\,i\;\mathrm{of}\;'
              r'\tilde A^{\!\top}_t = 0\;\Rightarrow\;'
              r'z^{\mathrm{thr}}_{t,i} = \tilde\varepsilon_{t,i} '
              r'= \varepsilon_{t-1,i} + d\varepsilon_i$',
              r'$\varepsilon_{t-1,i} \;=\; z_{t-1,i} - (A^{\!\top} z_{t-1})_i\;\;'
              r'(\mathrm{original\;SCM})$',
              r'$\Rightarrow\;\;z^{\mathrm{thr}}_{t,i} \;=\; z_{t-1,i} - (A^{\!\top} z_{t-1})_i '
              r'+ dz_i - (A^{\!\top}dz)_i'
              r'\;=\; z_{t-1,i}+dz_i - (A^{\!\top}(z_{t-1}+dz))_i.\;\blacksquare$'],
             fontsize=16, color='#0B3D91', dpi=300, line_spacing=1.55, width=12.0)

# Interpretation box
add_box(slide, 0.4, 5.7, 12.65, 1.5, fill=RGBColor(0xFD, 0xE9, 0xE7), line=RED)
add_text(slide, 0.6, 5.8, 12.3, 0.4,
         'Interpretation:', size=14, bold=True, color=RED)
add_text(slide, 0.6, 6.2, 12.3, 0.4,
         'dε was calibrated to the un-mutilated graph, then A was mutilated → '
         'the parental term is double-counted.',
         size=14, color=BLACK)
add_text(slide, 0.6, 6.65, 12.3, 0.4,
         'The bias vanishes only when A ≈ 0 — exactly the regime that hid the bug throughout training.',
         size=14, color=BLACK)


# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 8 — ε recovery + persistence
# ═══════════════════════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, 'Persistence — ε Recovery in the Mutilated Regime')

add_section_label(slide, 0.5, 1.05, 12, 'Lemma 4 (ε recovery inverts the do)')
add_eq(slide, 0.7, 1.45, 'eq_l4',
       r'$\varepsilon_{t+1}\;:=\;(I - M_t A^{\!\top})\,z_t \;=\; \tilde\varepsilon_t.$',
       fontsize=22, color='#0B3D91', width=6.5)
add_bullets(slide, 0.7, 2.4, 12, 1.1, [
    'Direct from the SCM equation at step t in the mutilated regime.',
    'Verified to numerical zero in test_recover_epsilon_u_inverts.',
], size=14)

add_section_label(slide, 0.5, 3.55, 12, 'Why use the mutilated graph for recovery?')
add_bullets(slide, 0.7, 3.95, 12, 1.7, [
    'The intervention persists into step t+1 (CRISPR knockout is not transient).',
    'A^T would mis-attribute the new value of program i to its old parents — '
    'the parents have been detached.',
    'Knockout clamp: x[g] = 0 for every g in knockout_set before computing the next dx.',
], size=14)

# Episode state mini-table
add_box(slide, 0.4, 5.85, 12.65, 1.35, fill=LIGHT_GRAY)
add_text(slide, 0.6, 5.95, 12.3, 0.4,
         'env.reset() clears the persistence state:', size=14, bold=True, color=DARK)
add_text(slide, 0.6, 6.4, 12.3, 0.4,
         'current_u  ← 0   ·   knockout_set ← ∅   ·   step_count ← 0',
         size=14, color=DARK, font='Consolas')
add_text(slide, 0.6, 6.85, 12.3, 0.3,
         '(see CausalWorldModelEnv.reset, rl/env.py)',
         size=12, italic=True, color=GRAY)


# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 9 — Why this fixes the diagnostic + scheme dispatch
# ═══════════════════════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, 'Why This Fixes the Diagnostic + Code Layout')

# Sparsity by construction
add_section_label(slide, 0.5, 1.05, 6, 'u is sparse by construction')
add_eq(slide, 0.7, 1.45, 'eq_sparse',
       r'$\sum_g L_{g,i}=1\;\Rightarrow\; L_{g,i}\sim O(1/G)\;\mathrm{unless}\;'
       r'g\;\mathrm{loads\;heavily\;on}\;i$',
       fontsize=15, color='#0B3D91', width=6.2)
add_bullets(slide, 0.7, 2.3, 6.0, 2.4, [
    'Single-gene action  u = L[g,:]  is sparse with no threshold.',
    'Two-gene combo u = L[g₁,:] + L[g₂,:] may saturate when both load on the same program.',
    'L inherits sparsity of D — adding L1 on D makes u even sharper.',
    'Diagnostic prints ‖u‖₀ at thresholds {0, 0.1, 0.5} + L1/L2 + programs_per_gene.',
], size=12)

# Scheme dispatch
add_section_label(slide, 6.85, 1.05, 6.0, 'Code: both schemes coexist behind a flag')
add_text(slide, 7.0, 1.45, 6.0, 0.4, 'configs/*.yaml', size=12, bold=True, color=GRAY)
add_text(slide, 7.0, 1.85, 6.0, 2.0,
         'intervention:\n'
         '  scheme: u_score        # or "threshold"\n'
         '  u_clip_max: 1.0\n'
         '  u_accumulate: true\n'
         '  knockout_clamp: true\n'
         '\n'
         'phase1:\n'
         '  lambda_D_sparse: 0.001  # L1 on entries of D',
         size=11, color=DARK, font='Consolas')
add_bullets(slide, 7.0, 4.0, 6.0, 1.5, [
    'rl/env.py — episode persistence (current_u, knockout_set).',
    'evaluation/eval_combinatorial.py — sequential combo with u carry.',
    'evaluation/run_perturbation_eval.py — single-pert eval.',
    'lambda_D_sparse = 0.001 — new L1 on D entries (gene-level sparsity).',
], size=12)

# Validation summary bar
add_box(slide, 0.4, 5.6, 12.65, 1.55, fill=RGBColor(0xE8, 0xF7, 0xEB), line=GREEN)
add_text(slide, 0.6, 5.7, 12.3, 0.4,
         'Validation — tests/test_u_intervention.py:',
         size=14, bold=True, color=GREEN)
add_text(slide, 0.6, 6.1, 12.3, 0.4,
         '7 / 7 invariants pass to numerical zero on a controlled toy model:',
         size=13, color=DARK)
add_text(slide, 0.6, 6.5, 12.3, 0.4,
         'u=0 identity  ·  u_i=1 pins target  ·  monotonicity  ·  '
         'Σ_g L_{g,i}=1  ·  ε recovery inverts  ·  knockout-clamp idempotent  ·  '
         'legacy threshold path unchanged',
         size=12, color=BLACK, font='Consolas')


# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 10 — Commands & next steps
# ═══════════════════════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, 'Running the Grid Search + u-score Training')

add_section_label(slide, 0.5, 1.05, 12,
                  '1.  Diagnose first — grid sweep on a 5 k-cell Replogle K562 subset')
add_box(slide, 0.5, 1.5, 12.4, 0.95, fill=DARK)
add_text(slide, 0.7, 1.6, 12.0, 0.4,
         '# from the repo root (scCausalRL/)',
         size=12, color=RGBColor(0xC0, 0xC0, 0xC0), font='Consolas')
add_text(slide, 0.7, 2.0, 12.0, 0.4,
         'sbatch scripts/train_phase1_grid.sbatch',
         size=15, color=WHITE, font='Consolas')

add_bullets(slide, 0.6, 2.6, 12.0, 1.4, [
    '4 × 2 = 8 runs, 40 epochs each, ~30 min/run → total ≈ 4 h.',
    'Outputs: results/grid_search/<run_id>/{checkpoint, edge_prob_histogram.png, summary.json}.',
    'Aggregate row per run in results/grid_search/summary.csv.',
], size=13)

add_section_label(slide, 0.5, 4.1, 12,
                  '2.  Promote winner + run full Phase-1 with the u-score scheme')
add_box(slide, 0.5, 4.55, 12.4, 1.55, fill=DARK)
add_text(slide, 0.7, 4.65, 12.0, 0.4,
         '# pick winning row, edit configs/replogle_u_score.yaml lambda_sparse / lambda_prior / prior_margin',
         size=11, color=RGBColor(0xC0, 0xC0, 0xC0), font='Consolas')
add_text(slide, 0.7, 5.0, 12.0, 0.4,
         'sbatch scripts/train_phase1_replogle_u_score.sbatch',
         size=15, color=WHITE, font='Consolas')
add_text(slide, 0.7, 5.4, 12.0, 0.4,
         '# verify the DAG escaped init and u is sparse',
         size=11, color=RGBColor(0xC0, 0xC0, 0xC0), font='Consolas')
add_text(slide, 0.7, 5.75, 12.0, 0.4,
         'python evaluation/dag_diagnostics.py --checkpoint checkpoints/phase1_best.pt '
         '--output_dir results/dag_diagnostics_u',
         size=12, color=WHITE, font='Consolas')

add_section_label(slide, 0.5, 6.25, 12, '3.  Unit tests (quick sanity)')
add_box(slide, 0.5, 6.65, 12.4, 0.55, fill=DARK)
add_text(slide, 0.7, 6.72, 12.0, 0.4,
         'pip install pytest && pytest tests/test_u_intervention.py -v',
         size=14, color=WHITE, font='Consolas')


# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 11 — Quick reference
# ═══════════════════════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, 'Quick Reference', color=DARKBLUE)

add_section_label(slide, 0.5, 1.05, 12, 'Files added / modified')
add_text(slide, 0.6, 1.5, 12.3, 4.2,
         'models/dag.py                       +  scm_intervene_u\n'
         'models/causal_vae.py                +  compute_loadings, intervene_gene_u, recover_epsilon_u\n'
         'rl/env.py                           dispatch + current_u + knockout_set\n'
         'training/train_vae.py               compute_loadings each epoch; prior_margin plumb\n'
         'training/train_joint.py             plumb intervention config into env\n'
         'evaluation/dag_diagnostics.py       +  analyze_u_sparsity (importable)\n'
         'evaluation/eval_combinatorial.py    scheme dispatch (threshold | u_score)\n'
         'evaluation/run_perturbation_eval.py scheme dispatch\n'
         'configs/{default,replogle,combined}.yaml  +  intervention: block\n'
         'configs/replogle_u_score.yaml       NEW — u-score full Phase-1 config\n'
         'scripts/run_phase1_grid.py          NEW — grid sweep driver\n'
         'scripts/train_phase1_grid.sbatch    NEW — Slurm wrapper for grid sweep\n'
         'scripts/train_phase1_replogle_u_score.sbatch  NEW — Slurm wrapper for u-score training\n'
         'tests/test_u_intervention.py        NEW — 7 invariant unit tests\n'
         'docs/architecture_u_score.md        NEW — full math write-up',
         size=12, color=DARK, font='Consolas')

add_section_label(slide, 0.5, 6.0, 12, 'Doc with full proofs')
add_text(slide, 0.6, 6.4, 12.3, 0.4,
         'docs/architecture_u_score.md  —  Lemmas 1, 2, 4 + Theorem 3 + Proposition 5 with full derivations.',
         size=14, color=BLUE, font='Consolas')


# ── save ───────────────────────────────────────────────────────────────────
prs.save(OUT)
print(f'Saved: {OUT}')
print(f'(Equation PNG cache: {EQ_DIR})')
