#!/usr/bin/env python3
"""
generate_architecture.py
========================
Generates docs/CerebroTwin_Architecture_Diagram.pdf with:
  1. Orthogonal non-overlapping arrow routing (fixing the 1A -> 1B collision bug).
  2. Clean numbered flow sequence: 1A, 1B -> 2 -> 3 -> 4 -> 5 <--- 6 (validation) -> 7.
  3. Clean ASCII text rendering (zero missing glyph squares).
  4. Updated PhysioNet v1.0.1 dataset citation in the legend.
  5. Live Streamlit Cloud deployment badge and link.

DO NOT OVERWRITE OR MODIFY PRESENTATION FILES IN docs/.
"""

import os
import sys
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.lib.colors import HexColor, white
from reportlab.pdfgen.canvas import Canvas

# --- Color palette ---
DARK_NAVY   = HexColor("#0A192F")  # Deep background
NAVY_CARD   = HexColor("#112240")  # Panel card surface
CARD_BORDER = HexColor("#233554")  # Neutral card border
DEEP_BLUE   = HexColor("#1B3A5C")
MID_BLUE    = HexColor("#2E6F95")
TEAL        = HexColor("#14B8A6")  # Clinical / dynamic signals
CYAN        = HexColor("#06B6D4")  # Preprocessing
SKY_BLUE    = HexColor("#38BDF8")  # Core physiological state
ORANGE      = HexColor("#F97316")  # ML forecasting engine
GREEN_ACC   = HexColor("#22C55E")  # Forecast outputs
RED_ACC     = HexColor("#EF4444")  # Ground truth validation
PURPLE      = HexColor("#A855F7")  # Clinician dashboard / simulator
WHITE       = white
LIGHT_GRAY  = HexColor("#CBD5E1")
MUTED_GRAY  = HexColor("#94A3B8")
GOLD        = HexColor("#F59E0B")

def _rounded_rect(c, x, y, w, h, r, fill_color, stroke_color=None, stroke_width=1):
    """Draw a rounded rectangle."""
    c.saveState()
    c.setFillColor(fill_color)
    if stroke_color:
        c.setStrokeColor(stroke_color)
        c.setLineWidth(stroke_width)
    else:
        c.setStrokeColor(fill_color)
        c.setLineWidth(0)
    c.roundRect(x, y, w, h, r, fill=1, stroke=1 if stroke_color else 0)
    c.restoreState()

def _draw_text_block(c, x, y, w, h, lines, font_size=8, color=WHITE, bold_first=True, font="Helvetica", line_spacing=None):
    """Center-align text lines inside a box area."""
    c.saveState()
    spacing = line_spacing if line_spacing is not None else (font_size + 3.5)
    total_h = len(lines) * spacing
    start_y = y + h / 2 + total_h / 2 - font_size
    for i, line in enumerate(lines):
        fn = f"{font}-Bold" if (bold_first and i == 0) else font
        c.setFont(fn, font_size)
        c.setFillColor(color if not (bold_first and i == 0) else WHITE)
        tw = c.stringWidth(line, fn, font_size)
        c.drawString(x + (w - tw) / 2, start_y - i * spacing, line)
    c.restoreState()

def _draw_orthogonal_arrow(c, points, color=CYAN, width=1.5, head_size=7):
    """
    Draw a multi-segment orthogonal arrow through a list of (x, y) points.
    Places a solid arrowhead at the terminal point pointing along the final segment.
    """
    if len(points) < 2:
        return
    c.saveState()
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.setLineCap(1)
    c.setLineJoin(1)
    
    p = c.beginPath()
    p.moveTo(points[0][0], points[0][1])
    for pt in points[1:]:
        p.lineTo(pt[0], pt[1])
    c.drawPath(p, fill=0, stroke=1)
    
    # Terminal arrowhead
    x_prev, y_prev = points[-2]
    x_end, y_end = points[-1]
    angle = math.atan2(y_end - y_prev, x_end - x_prev)
    
    c.setFillColor(color)
    c.setStrokeColor(color)
    p_head = c.beginPath()
    p_head.moveTo(x_end, y_end)
    p_head.lineTo(x_end - head_size * math.cos(angle - math.pi / 6),
                  y_end - head_size * math.sin(angle - math.pi / 6))
    p_head.lineTo(x_end - (head_size * 0.55) * math.cos(angle),
                  y_end - (head_size * 0.55) * math.sin(angle))
    p_head.lineTo(x_end - head_size * math.cos(angle + math.pi / 6),
                  y_end - head_size * math.sin(angle + math.pi / 6))
    p_head.close()
    c.drawPath(p_head, fill=1, stroke=0)
    c.restoreState()

def build_architecture_pdf(output_path: Path):
    """Build the production architecture diagram PDF."""
    W, H = landscape(LETTER)  # 792 x 612 pt
    c = Canvas(str(output_path), pagesize=landscape(LETTER))
    c.setTitle("CerebroTwin - Physiological Digital Twin Architecture")

    # 1. Full canvas background
    c.setFillColor(DARK_NAVY)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # 2. Header Title Bar
    title_y = H - 52
    title_h = 42
    _rounded_rect(c, 24, title_y, W - 48, title_h, 6, NAVY_CARD, CYAN, 1.5)
    c.saveState()
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 15)
    c.drawCentredString(W / 2, title_y + 24, "CerebroTwin  --  Physiological Digital Twin Architecture")
    c.setFont("Helvetica", 8.5)
    c.setFillColor(CYAN)
    c.drawCentredString(W / 2, title_y + 10, "Non-Cerebral Cardiovascular & Respiratory Biosignals  -->  0-60s Orthostatic MCAv Instability Forecasting")
    c.restoreState()

    # Layout Boundaries:
    # Main column: x = 24 to 586 (width = 562)
    # Legend column: x = 604 to 768 (width = 164)
    main_x = 24
    main_w = 562
    leg_x  = 604
    leg_w  = 164

    # =========================================================================
    # ROW 1: INPUTS & PREPROCESSING (y = 444 to 522, height = 78)
    # =========================================================================
    r1_y = 444
    r1_h = 78

    # 1A. Clinical Inputs
    w_1a = 162
    x_1a = main_x
    _rounded_rect(c, x_1a, r1_y, w_1a, r1_h, 8, NAVY_CARD, TEAL, 1.5)
    _draw_text_block(c, x_1a, r1_y, w_1a, r1_h, [
        "1A. Clinical Inputs",
        "Age | Sex | T2D Status",
        "HbA1c | Hypertension",
        "Medications (Antihypertensives)",
        "Resting MAP & Heart Rate",
    ], 7.5, LIGHT_GRAY)

    # 1B. Continuous Dynamic Signals
    w_1b = 180
    x_1b = x_1a + w_1a + 16  # 24 + 162 + 16 = 202
    _rounded_rect(c, x_1b, r1_y, w_1b, r1_h, 8, NAVY_CARD, TEAL, 1.5)
    _draw_text_block(c, x_1b, r1_y, w_1b, r1_h, [
        "1B. Continuous Dynamic Signals",
        "Beat-to-Beat Arterial BP (MAP)",
        "ECG / Heart Rate Dynamics",
        "Respiration Rate | EtCO2 Waveform",
        "0-5s Post-Stand Orthostatic Drop",
    ], 7.5, LIGHT_GRAY)

    # 2. Preprocessing & Alignment Engine
    w_2 = 188
    x_2 = main_x + main_w - w_2  # 24 + 562 - 188 = 398
    _rounded_rect(c, x_2, r1_y, w_2, r1_h, 8, NAVY_CARD, CYAN, 1.5)
    _draw_text_block(c, x_2, r1_y, w_2, r1_h, [
        "2. Preprocessing & Alignment",
        "Beat-to-Beat Synchronization",
        "Dynamic Window Alignment (0-60s)",
        "Signal QC & Outlier Imputation",
        "Z-score / Robust Feature Scaling",
    ], 7.5, LIGHT_GRAY)

    # --- BUG FIX: CLEAN NON-OVERLAPPING ORTHOGONAL ARROWS INTO BOX 2 ---
    # Arrow 1B -> 2: Straight horizontal from right of 1B into left of 2
    y_1b_mid = r1_y + r1_h / 2
    _draw_orthogonal_arrow(c, [
        (x_1b + w_1b, y_1b_mid),
        (x_2, y_1b_mid)
    ], color=TEAL, width=1.5, head_size=7)

    # Arrow 1A -> 2: Orthogonal routing ABOVE 1B (completely eliminates horizontal line bug)
    # Starts at top center of 1A, steps up into top channel (y = 534), runs across above 1B,
    # and drops directly into the top of Box 2.
    x_1a_top = x_1a + w_1a / 2
    x_2_top  = x_2 + 45
    y_top_bus = r1_y + r1_h + 10  # 444 + 78 + 10 = 532
    _draw_orthogonal_arrow(c, [
        (x_1a_top, r1_y + r1_h),
        (x_1a_top, y_top_bus),
        (x_2_top, y_top_bus),
        (x_2_top, r1_y + r1_h)
    ], color=TEAL, width=1.5, head_size=7)

    # Small label on top bus
    c.saveState()
    c.setFont("Helvetica-Bold", 6.5)
    c.setFillColor(TEAL)
    c.drawCentredString((x_1a_top + x_2_top) / 2, y_top_bus + 3, "Static Baseline Covariates Route (Non-Overlapping Orthogonal Flow)")
    c.restoreState()


    # =========================================================================
    # ROW 2: STATE CORE & ML FORECASTING ENGINE (y = 312 to 408, height = 96)
    # =========================================================================
    r2_y = 312
    r2_h = 96

    # 3. CerebroTwin Physiological State Core (27 Features)
    w_3 = 226
    x_3 = main_x
    _rounded_rect(c, x_3, r2_y, w_3, r2_h, 8, NAVY_CARD, SKY_BLUE, 1.8)
    _draw_text_block(c, x_3, r2_y, w_3, r2_h, [
        "3. CerebroTwin State Core (27 Features)",
        "Baroreflex Sensitivity Proxy (BRS)",
        "Cerebral Autoregulation Index (CAI)",
        "Delta HR/dt  |  Delta MAP/dt Slopes",
        "EtCO2 Vasoreactivity Gradients",
        "Autonomic Tone & Heart Rate Variability",
    ], 7.5, LIGHT_GRAY)

    # Arrow 2 -> 3: Orthogonal route from bottom of Box 2 to top of Box 3
    x_2_bot = x_2 + w_2 / 2
    x_3_top = x_3 + w_3 / 2
    y_bus_1_2 = r2_y + r2_h + 18  # 312 + 96 + 18 = 426
    _draw_orthogonal_arrow(c, [
        (x_2_bot, r1_y),
        (x_2_bot, y_bus_1_2),
        (x_3_top, y_bus_1_2),
        (x_3_top, r2_y + r2_h)
    ], color=CYAN, width=1.5, head_size=7)

    # 4. Multi-Target ML Forecasting Engine
    w_4 = 316
    x_4 = main_x + main_w - w_4  # 24 + 562 - 316 = 270
    _rounded_rect(c, x_4, r2_y, w_4, r2_h, 8, NAVY_CARD, ORANGE, 1.8)
    
    # Header inside Box 4
    c.saveState()
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawCentredString(x_4 + w_4 / 2, r2_y + r2_h - 14, "4. Multi-Target ML Forecasting Engine")
    c.setFont("Helvetica", 7)
    c.setFillColor(ORANGE)
    c.drawCentredString(x_4 + w_4 / 2, r2_y + r2_h - 25, "Ensemble of Progression Baselines & Production Quantile Gradient Boosting")
    c.restoreState()

    # Three model sub-cards inside Box 4
    sub_w = 94
    sub_h = 56
    sub_y = r2_y + 8
    models = [
        ("Model 0: Mean", "Heuristic Control Baseline", "Fixed Group Average", CARD_BORDER, MUTED_GRAY),
        ("Model 1: Ridge", "Linear Regularization", "L2 Penalty Shrinkage", CARD_BORDER, CYAN),
        ("Model 2: Quantile", "Multi-Output XGBoost", "90% Prediction Intervals", ORANGE, GOLD),
    ]
    for i, (m_title, m_desc1, m_desc2, m_border, m_accent) in enumerate(models):
        sx = x_4 + 8 + i * (sub_w + 8)
        _rounded_rect(c, sx, sub_y, sub_w, sub_h, 6, HexColor("#0D1B2E"), m_border, 1.2)
        _draw_text_block(c, sx, sub_y, sub_w, sub_h, [
            m_title,
            m_desc1,
            m_desc2,
        ], 6.5, m_accent, bold_first=True)

    # Arrow 3 -> 4: Straight horizontal connection between Core and ML Engine
    y_3_mid = r2_y + r2_h / 2
    _draw_orthogonal_arrow(c, [
        (x_3 + w_3, y_3_mid),
        (x_4, y_3_mid)
    ], color=SKY_BLUE, width=1.5, head_size=7)


    # =========================================================================
    # ROW 3: FORECAST OUTPUTS & VALIDATION GATE (y = 176 to 276, height = 100)
    # =========================================================================
    r3_y = 176
    r3_h = 100

    # 5. 0-60s MCAv Forecast Outputs
    w_5 = 265
    x_5 = main_x
    _rounded_rect(c, x_5, r3_y, w_5, r3_h, 8, NAVY_CARD, GREEN_ACC, 2)
    _draw_text_block(c, x_5, r3_y, w_5, r3_h, [
        "5. 0-60s MCAv Forecast Outputs",
        "(19 Orthostatic Targets + 90% Prediction Intervals)",
        "----------------------------------------------------------",
        "Predicted Peak Relative Drop (pct_mca_drop)",
        "Minimum Velocity / Nadir (MCAv_min cm/s) & Time-to-Nadir",
        "Cumulative Cerebral Hypoperfusion Deficit (AUC < Baseline)",
        "Trajectory Recovery Status: Rapid vs Delayed Compensator",
        "Clinical Alert: Normal vs High-Deviation Postural Drop",
    ], 7.2, LIGHT_GRAY)

    # Arrow 4 -> 5: Orthogonal route from Production Model (Model 2) down to Box 5
    x_4_out = x_4 + w_4 - sub_w / 2  # Model 2 center
    x_5_top = x_5 + w_5 / 2
    y_bus_2_3 = r3_y + r3_h + 18    # 176 + 100 + 18 = 294
    _draw_orthogonal_arrow(c, [
        (x_4_out, r2_y),
        (x_4_out, y_bus_2_3),
        (x_5_top, y_bus_2_3),
        (x_5_top, r3_y + r3_h)
    ], color=GREEN_ACC, width=1.5, head_size=7)

    # 6. Ground Truth Validation Gate
    w_6 = 265
    x_6 = main_x + main_w - w_6  # 24 + 562 - 265 = 321
    _rounded_rect(c, x_6, r3_y, w_6, r3_h, 8, HexColor("#1E1424"), RED_ACC, 2)
    _draw_text_block(c, x_6, r3_y, w_6, r3_h, [
        "6. Ground Truth Validation Gate",
        "(TCD MCAv Strictly Isolated from Runtime Inputs)",
        "----------------------------------------------------------",
        "Measured Transcranial Doppler (TCD) Ultrasound",
        "Zero-Leakage 5-Fold GroupKFold Subject Split",
        "Subject Exists in Train OR Test -- Never Both",
        "Primary Peak R2 = 0.735  |  Drop MAE = 4.816%",
        "Continuous 0-60s Trajectory RMSE = 4.475%",
    ], 7.2, HexColor("#FCA5A5"))

    # Dashed validation line connecting 6 (Validation) to 5 (Forecast Outputs)
    c.saveState()
    c.setStrokeColor(RED_ACC)
    c.setLineWidth(1.5)
    c.setDash(4, 3)
    c.line(x_6, r3_y + r3_h / 2, x_5 + w_5, r3_y + r3_h / 2)
    c.restoreState()
    
    # Validation comparison arrow head pointing at Box 5
    _draw_orthogonal_arrow(c, [
        (x_5 + w_5 + 18, r3_y + r3_h / 2),
        (x_5 + w_5, r3_y + r3_h / 2)
    ], color=RED_ACC, width=1.5, head_size=6)

    # Label on dashed validation line
    c.saveState()
    c.setFont("Helvetica-Bold", 6.5)
    c.setFillColor(RED_ACC)
    c.drawCentredString((x_5 + w_5 + x_6) / 2, r3_y + r3_h / 2 + 5, "GroupKFold Benchmark")
    c.restoreState()


    # =========================================================================
    # ROW 4: CLINICIAN DASHBOARD & SIMULATOR (y = 40 to 142, height = 102)
    # =========================================================================
    r4_y = 40
    r4_h = 102
    w_7  = main_w
    x_7  = main_x
    _rounded_rect(c, x_7, r4_y, w_7, r4_h, 8, NAVY_CARD, PURPLE, 2)
    _draw_text_block(c, x_7, r4_y, w_7, r4_h, [
        "7. Clinician Dashboard & Counterfactual What-If Simulator",
        "(Live Deployed on Streamlit Community Cloud)",
        "--------------------------------------------------------------------------------------------------------------------------------",
        "Interactive Patient Waveform Replay  |  Real-Time 0-60s MCAv Curve Overlay & 90% Confidence Ribbons",
        "Counterfactual Simulation: 'What if HbA1c drops to 7.0%?'  |  'What if Antihypertensive Dosage Adjusted?'",
        "SHAP Feature Importance Explanations (TreeExplainer Waterfall)  |  Orthostatic Syncope Deficit Risk Scoring",
        "Live Cloud App: https://cerebrotwin-cxdos3it3cqlvhu6eftmtt.streamlit.app/",
    ], 7.5, HexColor("#E9D5FF"))

    # Clean downward arrows from Box 5 and Box 6 directly down into Box 7
    y_bus_3_4 = r4_y + r4_h
    _draw_orthogonal_arrow(c, [
        (x_5 + w_5 / 2, r3_y),
        (x_5 + w_5 / 2, y_bus_3_4)
    ], color=GREEN_ACC, width=1.6, head_size=7)

    _draw_orthogonal_arrow(c, [
        (x_6 + w_6 / 2, r3_y),
        (x_6 + w_6 / 2, y_bus_3_4)
    ], color=RED_ACC, width=1.6, head_size=7)


    # =========================================================================
    # SIDEBAR COLUMN: LEGEND, VALIDATION METRICS & DATASET CITATION
    # =========================================================================
    # 1. Legend Card (y = 356 to 522, height = 166)
    leg_card_y = 356
    leg_card_h = 166
    _rounded_rect(c, leg_x, leg_card_y, leg_w, leg_card_h, 8, NAVY_CARD, CARD_BORDER, 1)

    c.saveState()
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(leg_x + 12, leg_card_y + leg_card_h - 16, "Architecture Legend")
    c.restoreState()

    legend_items = [
        (TEAL,      "1A/1B. Clinical & Signals"),
        (CYAN,      "2. Preprocessing & Sync"),
        (SKY_BLUE,  "3. Digital Twin State"),
        (ORANGE,    "4. Multi-Target ML"),
        (GREEN_ACC, "5. Forecast Outputs"),
        (RED_ACC,   "6. Ground Truth Isolation"),
        (PURPLE,    "7. Clinician Dashboard"),
    ]
    for i, (clr, label) in enumerate(legend_items):
        item_y = leg_card_y + leg_card_h - 35 - i * 18
        # Color badge
        _rounded_rect(c, leg_x + 12, item_y - 2, 12, 12, 2, clr, None)
        c.saveState()
        c.setFont("Helvetica", 7.2)
        c.setFillColor(LIGHT_GRAY)
        c.drawString(leg_x + 30, item_y + 1, label)
        c.restoreState()

    # 2. Validation Metrics Card (y = 236 to 344, height = 108)
    met_y = 236
    met_h = 108
    _rounded_rect(c, leg_x, met_y, leg_w, met_h, 8, NAVY_CARD, CARD_BORDER, 1)
    _draw_text_block(c, leg_x, met_y, leg_w, met_h, [
        "Validation Performance",
        "5-Fold Subject GroupKFold",
        "-------------------------------",
        "Primary Peak R2: 0.735",
        "Drop MAE: 4.816%",
        "Trajectory RMSE: 4.475%",
        "Uncertainty: 90% Quantile CI",
    ], 7.5, SKY_BLUE)

    # 3. PhysioNet Dataset Citation Card (y = 40 to 224, height = 184)
    # EXACT user specification:
    # "PhysioNet Dataset: Cerebral Perfusion & Vasoregulation in Type 2 Diabetes (v1.0.1) -- physionet.org/content/cerebral-perfusion-diabetes/1.0.1/"
    ds_y = 40
    ds_h = 184
    _rounded_rect(c, leg_x, ds_y, leg_w, ds_h, 8, NAVY_CARD, CARD_BORDER, 1)
    _draw_text_block(c, leg_x, ds_y, leg_w, ds_h, [
        "PhysioNet Dataset",
        "Cerebral Perfusion &",
        "Vasoregulation in",
        "Type 2 Diabetes (v1.0.1)",
        "------------------------------------",
        "Primary Repository URL:",
        "physionet.org/content/",
        "cerebral-perfusion-",
        "diabetes/1.0.1/",
        "------------------------------------",
        "Subject Cohort: T2D & Controls",
        "Continuous TCD, ABP, ECG, EtCO2",
        "Zero Pre-Stand Runtime Leakage",
    ], 7, TEAL)


    # =========================================================================
    # FOOTER
    # =========================================================================
    c.saveState()
    c.setFillColor(MUTED_GRAY)
    c.setFont("Helvetica", 7)
    c.drawCentredString(W / 2, 14, "CerebroTwin (c) 2026 Harish Pannuru, Pavan Sai Krishna, Girish Chandra Kovuru  --  IIITDM Kancheepuram  --  All Rights Reserved")
    c.restoreState()

    c.showPage()
    c.save()
    print(f"  [OK] Architecture diagram generated successfully -> {output_path}")

if __name__ == "__main__":
    arch_path = DOCS_DIR / "CerebroTwin_Architecture_Diagram.pdf"
    print("Generating updated CerebroTwin Architecture Diagram...")
    build_architecture_pdf(arch_path)
    
    if arch_path.exists() and arch_path.stat().st_size > 0:
        print(f"  Verified: {arch_path} ({arch_path.stat().st_size:,} bytes)")
    else:
        print(f"  FAILED to generate: {arch_path}")
        sys.exit(1)
