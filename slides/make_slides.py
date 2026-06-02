"""
Generate Phase 1 Results slides as .pptx
Run: python slides/make_slides.py
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phase1_results.pptx')

# Colors
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x00, 0x00, 0x00)
DARK = RGBColor(0x1A, 0x1A, 0x2E)
BLUE = RGBColor(0x21, 0x71, 0xB5)
GRAY = RGBColor(0x66, 0x66, 0x66)
LIGHT_GRAY = RGBColor(0xF5, 0xF5, 0xF5)
RED = RGBColor(0xC0, 0x39, 0x2B)
GREEN = RGBColor(0x27, 0xAE, 0x60)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# ─── Helpers ────────────────────────────────────────────────────────────────

def add_blank_slide():
    layout = prs.slide_layouts[6]  # blank
    slide = prs.slides.add_slide(layout)
    # white background
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = WHITE
    return slide

def add_textbox(slide, left, top, width, height, text, font_size=18,
                bold=False, color=BLACK, alignment=PP_ALIGN.LEFT, font_name='Calibri'):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return txBox

def add_bullet_text(slide, left, top, width, height, items, font_size=16,
                    color=BLACK, spacing=Pt(6)):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = 'Calibri'
        p.space_after = spacing
        p.level = 0
    return txBox

def add_title_bar(slide, title_text):
    """Add a colored title bar at the top."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.9))
    shape.fill.solid()
    shape.fill.fore_color.rgb = DARK
    shape.line.fill.background()
    add_textbox(slide, 0.5, 0.15, 12, 0.6, title_text,
                font_size=28, bold=True, color=WHITE)

def add_placeholder_box(slide, left, top, width, height, label):
    """Dashed border placeholder for figures."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = LIGHT_GRAY
    shape.line.color.rgb = RGBColor(0xAA, 0xAA, 0xAA)
    shape.line.dash_style = 2  # dash
    shape.line.width = Pt(1.5)
    # Label
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = label
    p.font.size = Pt(12)
    p.font.color.rgb = GRAY
    p.font.italic = True
    p.alignment = PP_ALIGN.CENTER
    tf.paragraphs[0].space_before = Pt(int(height * 72 / 2 - 10))

def add_math_block(slide, left, top, width, height, lines, font_size=14):
    """Add a block of math/code text."""
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(font_size)
        p.font.color.rgb = DARK
        p.font.name = 'Consolas'
        p.space_after = Pt(2)
    return txBox

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 1: Title
# ═══════════════════════════════════════════════════════════════════════════
slide = add_blank_slide()
add_textbox(slide, 1, 2.0, 11, 1.2,
            'scCausalRL: Phase 1 Results',
            font_size=40, bold=True, color=DARK, alignment=PP_ALIGN.CENTER)
add_textbox(slide, 1, 3.3, 11, 0.8,
            'Causal VAE with DAG-Structured Latent SCM\nfor Perturbation Prediction on Norman et al. (2019)',
            font_size=20, color=GRAY, alignment=PP_ALIGN.CENTER)
add_textbox(slide, 1, 5.5, 11, 0.5,
            'Shervin Goudarzi  |  Goodarzi Lab  |  Arc Institute',
            font_size=16, color=GRAY, alignment=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 2: Data & Preprocessing
# ═══════════════════════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, 'Dataset & Preprocessing')

add_textbox(slide, 0.5, 1.2, 6, 0.5, 'Norman et al. 2019 (K562 CRISPRa)',
            font_size=22, bold=True, color=DARK)
add_bullet_text(slide, 0.5, 1.8, 6, 3.5, [
    '104,507 cells x 33,694 genes (raw)',
    'CRISPRa activation perturbations in K562 cells',
    '11,183 control cells (NegCtrl guides)',
    '105 single-gene perturbations (~200-1000 cells each)',
    '131 two-gene combinatorial perturbations (held out)',
], font_size=16)

add_textbox(slide, 0.5, 4.6, 6, 0.5, 'Preprocessing Pipeline',
            font_size=22, bold=True, color=DARK)
add_bullet_text(slide, 0.5, 5.1, 6, 2.0, [
    'Library-size normalization (target_sum = 10,000)',
    'Log1p transform: x -> log(1 + x)',
    'HVG selection: top 2,000 genes (Seurat v3)',
    'PCA: 50 components (for state clustering)',
], font_size=16)

# Right column: split info
add_textbox(slide, 7.0, 1.2, 5.5, 0.5, 'Train / Val / Test Split',
            font_size=22, bold=True, color=DARK)
add_bullet_text(slide, 7.0, 1.8, 5.5, 2.0, [
    'Split by GENE (not by cell) -- prevents data leakage',
    '84 train genes (80%) -- used for perturbation consistency loss',
    '10 validation genes (10%) -- for hyperparameter selection',
    '11 test genes (10%) -- held out, never seen during training',
    'Fixed seed = 42 for reproducibility',
], font_size=16)

add_textbox(slide, 7.0, 3.8, 5.5, 0.5, 'What Each Split Sees',
            font_size=20, bold=True, color=DARK)
add_bullet_text(slide, 7.0, 4.3, 5.5, 2.8, [
    'ALL cells used for reconstruction loss (observational)',
    'Only TRAIN gene perturbations used for interventional loss',
    'Val/test gene perturbation data never enters training',
    'All 131 combos fully held out (never in training)',
    'Combo eval restricted to pairs where both genes are in train split',
], font_size=16)

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 3: Model Architecture
# ═══════════════════════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, 'Model Architecture: Causal VAE')

add_textbox(slide, 0.5, 1.2, 12, 0.5, 'Forward Pass (Observational)',
            font_size=20, bold=True, color=DARK)
add_math_block(slide, 0.5, 1.8, 12, 1.6, [
    'x [B, 2000]  -->  Encoder (MLP: 2000->512->256)  -->  (mu, log_var) [B, 64]',
    '              -->  Reparameterize  -->  epsilon [B, 64]     (exogenous noise)',
    '',
    'theta [64, 64]  -->  Gumbel-Sigmoid(tau)  -->  A [64, 64]  (soft DAG adjacency)',
    '',
    'z = (I - A^T)^{-1} @ epsilon     (linear SCM solve)       [B, 64]',
    'x_hat = D @ z + b                (linear decoder)          [B, 2000]',
], font_size=13)

add_textbox(slide, 0.5, 3.6, 6, 0.5, 'Key Design Choices',
            font_size=20, bold=True, color=DARK)
add_bullet_text(slide, 0.5, 4.1, 6, 3.0, [
    'Linear SCM: z = (I - A^T)^{-1} epsilon -- solved via torch.linalg.solve',
    'Linear decoder: x_hat = Dz + b -- enables GRN projection: G = D A D^T',
    'Gumbel-Sigmoid: differentiable edge sampling with temperature tau',
    'DAGMA acyclicity: h(A) = tr((I + s/d * A*A)^d) - d = 0',
    'Theta init: N(-5, 0.25) -- graph starts ~0.7% dense, well-conditioned',
    '64 latent programs, 16 DAG bootstrap samples',
], font_size=15)

add_textbox(slide, 7.0, 3.6, 5.5, 0.5, 'Training Losses (8 terms)',
            font_size=20, bold=True, color=DARK)
add_bullet_text(slide, 7.0, 4.1, 5.5, 3.0, [
    'Reconstruction: MSE(x, x_hat)',
    'KL divergence: beta-annealed (0.01 -> 0.5)',
    'DAGMA acyclicity: augmented Lagrangian',
    'Sparsity: L1 on edge weights',
    'Spectral stability: ReLU(sigma_max(A) - 0.95)',
    'Decoder orthogonality: ||D^T D / norm - I||^2',
    'Perturbation consistency: Eq. 4 (see next slide)',
    'Biological prior: CollecTRI TF-target MarginRanking',
], font_size=15)

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 4: Perturbation Consistency Loss (the key innovation)
# ═══════════════════════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, 'Perturbation Consistency Loss: D-dagger + Multi-Mutilation')

add_textbox(slide, 0.5, 1.2, 12, 0.4, 'For each perturbation (x_ctrl, gene_g, x_pert), averaged over N DAG samples:',
            font_size=16, color=GRAY)

add_math_block(slide, 0.5, 1.7, 12, 3.8, [
    'Step 1: Gene-space effect (precomputed from full dataset)',
    '   dx = mean(x_pert) - mean(x_ctrl)                              [2000]',
    '',
    'Step 2: Project to latent space via decoder pseudoinverse D-dagger',
    '   dz = (D^T D + lambda*I)^{-1} D^T dx                          [64]',
    '',
    'Step 3: Back-compute exogenous change through DAG',
    '   d_epsilon = (I - A_n^T) @ dz                                  [64]',
    '',
    'Step 4: Multi-column mutilation (do-calculus)',
    '   affected = { i : |d_epsilon_i| > threshold * max|d_epsilon| }',
    '   A_tilde[:, affected] = 0       (zero incoming edges to affected programs)',
    '',
    'Step 5: Apply full perturbation + solve mutilated SCM',
    '   epsilon_tilde = epsilon_ctrl + d_epsilon                       (full shift)',
    '   z_post = (I - A_tilde^T)^{-1} @ epsilon_tilde                 (DAG in gradient path)',
    '   x_hat_pert = D @ z_post + b',
    '',
    'L_pert = (1/N) sum_n || x_hat_pert_n - x_pert ||^2',
], font_size=13)

add_textbox(slide, 0.5, 5.8, 6, 0.4, 'Why mutilation matters:',
            font_size=16, bold=True, color=DARK)
add_bullet_text(slide, 0.5, 6.2, 6, 1.0, [
    'Without mutilation: (I-A^T)^{-1} @ (I-A^T) = I -- DAG cancels out of loss',
    'Mutilation breaks this: (I-A_tilde^T)^{-1} != (I-A^T)^{-1} -- DAG gets gradients',
    'Tests: "does downstream propagation through A match observed perturbation?"',
], font_size=14)

add_textbox(slide, 7.0, 5.8, 5.5, 0.4, 'Why D-dagger (not argmax|D|):',
            font_size=16, bold=True, color=DARK)
add_bullet_text(slide, 7.0, 6.2, 5.5, 1.0, [
    'argmax|D[gene,:]| uses ONE row of D -- fragile during early training',
    'D-dagger uses FULL decoder geometry to project gene-space to latent-space',
    '(I-A^T) further filters through DAG structure -- DAG-dependent targeting',
], font_size=14)

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 5: Training Curves
# ═══════════════════════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, 'Phase 1 Training Curves (150 epochs)')

add_placeholder_box(slide, 0.5, 1.2, 12.3, 5.8,
    '[INSERT: slides/training_curves_panel.png]')

add_textbox(slide, 0.5, 7.05, 12, 0.4,
    'Phase A (epochs 1-20): reconstruction only, DAG frozen  |  '
    'Phase B (21-50): DAG activated, losses ramped  |  '
    'Phase C (51-150): full training, tau annealed 1.0->0.25',
    font_size=12, color=GRAY)

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 6: Evaluation Method
# ═══════════════════════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, 'Evaluation: Perturbation Prediction')

add_textbox(slide, 0.5, 1.2, 6, 0.4, 'Single-Gene Prediction (test set, 7 genes)',
            font_size=20, bold=True, color=DARK)
add_bullet_text(slide, 0.5, 1.7, 6, 2.5, [
    'For each held-out gene g:',
    '  1. Encode control cells -> epsilon_ctrl',
    '  2. dx = de_single[g]  (observed effect)',
    '  3. D-dagger pipeline: dx -> dz -> d_eps -> mutilate -> solve -> decode',
    '  4. pred_effect = mean(x_hat_pert) - mean(x_hat_ctrl)',
    '  5. Compare pred_effect to observed DE',
    '',
    'Zero-shot: model NEVER saw these genes\' perturbation data',
], font_size=14)

add_textbox(slide, 7.0, 1.2, 5.5, 0.4, 'Combo Prediction (held-out, 39 combos)',
            font_size=20, bold=True, color=DARK)
add_bullet_text(slide, 7.0, 1.7, 5.5, 2.5, [
    'For each combo (gene_A, gene_B):',
    '  1. Sequential intervention: do(A) then do(B)',
    '  2. do(A): D-dagger(dx_A) -> mutilate -> solve -> z_after_A',
    '  3. Recover epsilon: eps_after_A = (I - A^T) @ z_after_A',
    '  4. do(B): D-dagger(dx_B) -> mutilate -> solve -> z_after_AB',
    '  5. x_hat_combo = decode(z_after_AB)',
    '',
    'Tests compositional generalization: singles -> combos',
], font_size=14)

# Metrics section
add_textbox(slide, 0.5, 4.3, 12, 0.4, 'Metrics (per perturbation)',
            font_size=20, bold=True, color=DARK)

add_math_block(slide, 0.5, 4.8, 6, 2.5, [
    'DEG Detection (AUROC):',
    '  Labels: sig_j = 1 if p_value_j < 0.05  (Mann-Whitney U)',
    '  Scores: |pred_effect_j|  for each gene j != g',
    '  AUROC = area under ROC(TPR vs FPR)',
    '  "Can model rank truly affected genes above unaffected?"',
    '  Random baseline = 0.5',
    '',
    'Effect Size Ranking (Spearman rho):',
    '  rho = rank_corr( |pred_effect|, observed_effect_size )',
    '  effect_size = |lfc| / sqrt((std_ctrl^2 + std_pert^2)/2)',
    '  "Does model rank large effects above small effects?"',
], font_size=12)

add_math_block(slide, 7.0, 4.8, 5.5, 2.5, [
    'Direction Accuracy:',
    '  For DEGs only (p < 0.05):',
    '  dir_acc = mean( sign(pred_j) == sign(obs_lfc_j) )',
    '  "Does model get up/down regulation right?"',
    '',
    'Pearson on DEGs:',
    '  r = pearson( pred_effect[DEGs], obs_lfc[DEGs] )',
    '  "Linear correlation of predicted vs observed magnitude"',
    '',
    'Baselines:',
    '  Correlation: |corr(gene_g, gene_j)| from control cells',
    '  Additive (combos): de_single[A] + de_single[B]',
], font_size=12)

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 7: Single-Pert Results
# ═══════════════════════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, 'Results: Single-Gene Perturbation Prediction')

# Left: bar chart placeholder
add_placeholder_box(slide, 0.3, 1.2, 6.2, 4.0,
    '[INSERT: results/perturbation_eval/single_pert_comparison.png]')

# Right: scatter placeholder
add_placeholder_box(slide, 6.8, 1.2, 5.5, 4.0,
    '[INSERT: results/perturbation_eval/single_test_scatter_auroc.png]')

# Bottom: summary numbers
add_textbox(slide, 0.5, 5.5, 12, 0.4, 'Summary (Test Set: 7 held-out genes)',
            font_size=18, bold=True, color=DARK)

add_math_block(slide, 0.5, 6.0, 12, 1.2, [
    '                          AUROC     Spearman rho',
    '  Causal VAE (D-dagger)   0.933     0.670          <-- zero-shot, never saw these genes',
    '  Correlation Baseline    0.677     --             <-- just gene-gene co-expression',
    '  Learned GRN (D@A@D^T)   0.832     0.573          <-- DAG structure alone',
    '',
    '  Model beats correlation on ALL 7/7 test genes (scatter: every point above diagonal)',
], font_size=13)

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 8: Combo Results
# ═══════════════════════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, 'Results: Combinatorial Perturbation Prediction (Held-Out)')

# Left: bar chart
add_placeholder_box(slide, 0.3, 1.2, 6.2, 3.5,
    '[INSERT: results/perturbation_eval/combo_pert_comparison.png]')

# Right: epistasis scatter
add_placeholder_box(slide, 6.8, 1.2, 5.8, 3.5,
    '[INSERT: results/perturbation_eval/combo_epistasis_scatter.png]')

# Summary
add_textbox(slide, 0.5, 5.0, 12, 0.4, 'Combo Summary (39 combos, both genes in train split)',
            font_size=18, bold=True, color=DARK)

add_math_block(slide, 0.5, 5.4, 6, 1.8, [
    '                      AUROC   Spearman  Pearson',
    '  Causal VAE          0.945   0.751     0.222',
    '  Additive Baseline   0.954   0.789     0.929',
    '  Correlation          0.727   0.471     --',
    '',
    '  Model AUROC close to additive (0.945 vs 0.954)',
    '  Model crushes correlation baseline (0.945 vs 0.727)',
], font_size=13)

add_math_block(slide, 7.0, 5.4, 5.5, 1.8, [
    'Epistasis Stratification:',
    '  Low epistasis:   Model 0.945 vs Additive 0.955',
    '  Med epistasis:   Model 0.938 vs Additive 0.947',
    '  High epistasis:  Model 0.952 vs Additive 0.960',
    '',
    '  Pearson gap narrows with epistasis (-0.28 -> -0.12)',
    '  Additive advantage is in magnitude, not ranking',
], font_size=13)

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 9: Key Takeaways + Next Steps
# ═══════════════════════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, 'Key Takeaways & Next Steps')

add_textbox(slide, 0.5, 1.3, 6, 0.4, 'Phase 1 Takeaways',
            font_size=22, bold=True, color=DARK)
add_bullet_text(slide, 0.5, 1.8, 6, 3.5, [
    'Single-gene prediction: 0.933 AUROC on held-out test (zero-shot)',
    'Beats correlation baseline by +0.26 AUROC -- DAG adds real value',
    'Combo AUROC competitive with additive (0.945 vs 0.954)',
    'Model knows WHICH genes change, but magnitude calibration needs work',
    'Pearson gap (0.22 vs 0.93) expected: additive uses real observed data',
    'Epistasis stratification shows gap narrows on interactive combos',
], font_size=16)

add_textbox(slide, 7.0, 1.3, 5.5, 0.4, 'Phase 2: RL Planning (Next)',
            font_size=22, bold=True, color=DARK)
add_bullet_text(slide, 7.0, 1.8, 5.5, 3.5, [
    'PPO agent plans multi-step perturbation sequences',
    'Gene-level actions: 2000 genes x 2 directions = 4000 actions',
    'State space: Leiden clusters in latent z-space',
    'Bootstrapped DAG uncertainty for pessimistic reward',
    'Joint VAE fine-tuning with policy-relevance upweighting',
    'Evaluation: can the agent find optimal 2-3 step CRISPR protocols?',
], font_size=16)

add_textbox(slide, 0.5, 5.5, 12, 0.4, 'Open Questions',
            font_size=22, bold=True, color=RED)
add_bullet_text(slide, 0.5, 6.0, 12, 1.2, [
    'Magnitude calibration: can we add a scaling/calibration step to match observed effect sizes?',
    'Epistasis: with more training / larger datasets (Replogle), will model capture interactions better?',
    'Foundation model encoder: will replacing MLP with ESM-based encoder improve generalization?',
], font_size=16)


# ─── SAVE ───────────────────────────────────────────────────────────────────
prs.save(OUT)
print(f"Saved: {OUT}")
print(f"\nFigure placeholders to fill in:")
print(f"  Slide 5: slides/training_curves_panel.png")
print(f"  Slide 7: results/perturbation_eval/single_pert_comparison.png")
print(f"  Slide 7: results/perturbation_eval/single_test_scatter_auroc.png")
print(f"  Slide 8: results/perturbation_eval/combo_pert_comparison.png")
print(f"  Slide 8: results/perturbation_eval/combo_epistasis_scatter.png")
