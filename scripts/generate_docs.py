#!/usr/bin/env python3
"""
generate_docs.py
================
Self-contained script that generates two professional PDF documents
for the CerebroTwin hackathon submission:

  1. docs/CerebroTwin_Architecture_Diagram.pdf
  2. docs/CerebroTwin_Project_Presentation.pdf

Uses reportlab for precise layout control.
"""
import os
import sys
from pathlib import Path

# ── ensure project root ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import (
    HexColor, white, black, Color,
)
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Frame, PageTemplate, BaseDocTemplate, Flowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ─── Color palette ────────────────────────────────────────────────
DARK_NAVY   = HexColor("#0D1B2A")
NAVY        = HexColor("#1B2838")
DEEP_BLUE   = HexColor("#1B3A5C")
MID_BLUE    = HexColor("#2E6F95")
TEAL        = HexColor("#1A8A7D")
ACCENT      = HexColor("#4FC3F7")
LIGHT_GRAY  = HexColor("#E8EDF2")
WHITE       = white
ORANGE      = HexColor("#FF8A65")
PURPLE      = HexColor("#AB47BC")
GREEN_ACC   = HexColor("#66BB6A")
RED_ACC     = HexColor("#EF5350")

# ═══════════════════════════════════════════════════════════════════
# PART 1:  Architecture Diagram (single-page landscape PDF)
# ═══════════════════════════════════════════════════════════════════

def _rounded_rect(c, x, y, w, h, r, fill_color, stroke_color=None, stroke_width=1):
    """Draw a rounded rectangle on the canvas."""
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


def _draw_text_block(c, x, y, w, h, lines, font_size=8, color=WHITE, bold_first=True, font="Helvetica"):
    """Centre-align a list of text lines inside a box area."""
    c.saveState()
    c.setFillColor(color)
    total_h = len(lines) * (font_size + 3)
    start_y = y + h / 2 + total_h / 2 - font_size
    for i, line in enumerate(lines):
        fn = f"{font}-Bold" if (bold_first and i == 0) else font
        c.setFont(fn, font_size)
        tw = c.stringWidth(line, fn, font_size)
        c.drawString(x + (w - tw) / 2, start_y - i * (font_size + 3), line)
    c.restoreState()


def _arrow(c, x1, y1, x2, y2, color=ACCENT, width=1.5):
    """Draw a simple arrow line."""
    c.saveState()
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.setLineCap(1)
    c.line(x1, y1, x2, y2)
    # arrowhead
    import math
    angle = math.atan2(y2 - y1, x2 - x1)
    arrow_len = 8
    c.line(x2, y2,
           x2 - arrow_len * math.cos(angle - math.pi / 6),
           y2 - arrow_len * math.sin(angle - math.pi / 6))
    c.line(x2, y2,
           x2 - arrow_len * math.cos(angle + math.pi / 6),
           y2 - arrow_len * math.sin(angle + math.pi / 6))
    c.restoreState()


def build_architecture_pdf(output_path: Path):
    """Build the architecture diagram PDF."""
    W, H = landscape(LETTER)  # 792 x 612
    c = Canvas(str(output_path), pagesize=landscape(LETTER))
    c.setTitle("CerebroTwin – Architecture Diagram")

    # Background
    c.setFillColor(DARK_NAVY)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # Title bar
    _rounded_rect(c, 20, H - 60, W - 40, 45, 6, DEEP_BLUE, ACCENT, 1.5)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 18)
    title = "CerebroTwin  —  Physiological Digital Twin Architecture"
    c.drawCentredString(W / 2, H - 44, title)

    # ── ROW 1: Inputs (y ≈ 470) ──
    row1_y = H - 145
    bw, bh = 165, 75

    # Box 1: Clinical Inputs
    x1 = 30
    _rounded_rect(c, x1, row1_y, bw, bh, 8, DEEP_BLUE, TEAL, 1.5)
    _draw_text_block(c, x1, row1_y, bw, bh, [
        "Clinical Inputs",
        "Age · T2D Status · HbA1c",
        "HTN · Medications",
        "Baseline HRV / BP",
    ], 8)

    # Box 2: Continuous Dynamic Signals
    x2 = 220
    _rounded_rect(c, x2, row1_y, bw + 10, bh, 8, DEEP_BLUE, TEAL, 1.5)
    _draw_text_block(c, x2, row1_y, bw + 10, bh, [
        "Continuous Dynamic Signals",
        "ECG / HR  ·  Arterial BP",
        "Respiration  ·  EtCO₂",
    ], 8)

    # Box 3: Preprocessing
    x3 = 425
    _rounded_rect(c, x3, row1_y, bw, bh, 8, MID_BLUE, ACCENT, 1.5)
    _draw_text_block(c, x3, row1_y, bw, bh, [
        "Preprocessing &",
        "Alignment Engine",
        "Beat-to-beat sync",
        "Normalization · QC",
    ], 8)

    # Arrows row 1 → preprocessing
    mid1 = row1_y + bh / 2
    _arrow(c, x1 + bw, mid1, x3, mid1)
    _arrow(c, x2 + bw + 10, mid1, x3, mid1)

    # ── ROW 2: Core engine (y ≈ 350) ──
    row2_y = H - 255
    core_w = 220

    # CerebroTwin Core
    x4 = 30
    _rounded_rect(c, x4, row2_y, core_w, 80, 10, HexColor("#1A3D5C"), ACCENT, 2)
    _draw_text_block(c, x4, row2_y, core_w, 80, [
        "CerebroTwin",
        "Physiological State Core",
        "Baroreflex · CA Index",
        "ΔHR/Δt · ΔMAP/Δt · CO₂ reactivity",
    ], 8.5)

    _arrow(c, x3 + bw / 2, row1_y, x4 + core_w / 2, row2_y + 80, ACCENT)

    # ML Forecasting Engine
    ml_x = 290
    ml_w = 280
    _rounded_rect(c, ml_x, row2_y, ml_w, 80, 10, HexColor("#1A3D5C"), ORANGE, 2)
    c.saveState()
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(ml_x + ml_w / 2, row2_y + 65, "Multi-Target ML Forecasting Engine")
    c.restoreState()

    # Three sub-boxes inside ML box
    sub_w = 80
    sub_h = 40
    sub_y = row2_y + 8
    models = [
        ("Model 0", "Mean Baseline", GREEN_ACC),
        ("Model 1", "Ridge (L2)", ACCENT),
        ("Model 2", "Quantile XGB", ORANGE),
    ]
    for i, (name, desc, clr) in enumerate(models):
        sx = ml_x + 10 + i * (sub_w + 10)
        _rounded_rect(c, sx, sub_y, sub_w, sub_h, 5, NAVY, clr, 1.5)
        _draw_text_block(c, sx, sub_y, sub_w, sub_h, [name, desc], 7, clr)

    _arrow(c, x4 + core_w, row2_y + 40, ml_x, row2_y + 40, ACCENT)

    # ── ROW 3: Outputs (y ≈ 220) ──
    row3_y = H - 380
    out_w = 240

    # Output trajectory box
    ox = 30
    _rounded_rect(c, ox, row3_y, out_w, 90, 10, HexColor("#0A2540"), GREEN_ACC, 2)
    _draw_text_block(c, ox, row3_y, out_w, 90, [
        "0–60s MCA Blood-Flow Velocity",
        "Trajectory Output",
        "─────────────────────",
        "Nadir (MCAv_min)  ·  Flow Deficit",
        "Recovery Classification",
        "90% Prediction Intervals",
    ], 8, GREEN_ACC)

    _arrow(c, ml_x + ml_w / 2, row2_y, ox + out_w / 2, row3_y + 90, GREEN_ACC)

    # Ground Truth Validation Gate
    gt_x = 310
    gt_w = 220
    _rounded_rect(c, gt_x, row3_y, gt_w, 90, 10, HexColor("#3E1A1A"), RED_ACC, 2)
    _draw_text_block(c, gt_x, row3_y, gt_w, 90, [
        "Ground Truth Validation Gate",
        "─────────────────────",
        "TCD MCAv (Transcranial Doppler)",
        "Strictly isolated from",
        "runtime model inputs",
        "GroupKFold Subject Split",
    ], 8, RED_ACC)

    # dashed validation arrow
    c.saveState()
    c.setStrokeColor(RED_ACC)
    c.setLineWidth(1.5)
    c.setDash(4, 3)
    c.line(ox + out_w, row3_y + 45, gt_x, row3_y + 45)
    c.restoreState()

    # ── ROW 4: Clinician Dashboard (y ≈ 100) ──
    row4_y = H - 500
    dash_w = 540
    dash_x = 30
    _rounded_rect(c, dash_x, row4_y, dash_w, 85, 10, NAVY, PURPLE, 2)
    _draw_text_block(c, dash_x, row4_y, dash_w, 85, [
        "Clinician Dashboard  &  Counterfactual What-If Simulator",
        "──────────────────────────────────────────",
        "Interactive Streamlit UI  ·  Patient Replay  ·  SHAP Feature Waterfall",
        "\"What if HbA1c = 7.5?\"  ·  \"What if β-blocker added?\"  ·  Trajectory comparison overlay",
    ], 9, PURPLE)

    _arrow(c, ox + out_w / 2, row3_y, dash_x + dash_w / 4, row4_y + 85, PURPLE)
    _arrow(c, gt_x + gt_w / 2, row3_y, dash_x + dash_w * 3 / 4, row4_y + 85, PURPLE)

    # ── Legend / Key  ──
    legend_x = 610
    legend_y = H - 145
    _rounded_rect(c, legend_x, legend_y - 270, 165, 330, 8, NAVY, HexColor("#555555"), 1)
    c.saveState()
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(legend_x + 10, legend_y + 47, "Legend")
    c.setFont("Helvetica", 7.5)

    items = [
        (TEAL, "Clinical / Signal Input"),
        (ACCENT, "Preprocessing / Linear"),
        (ORANGE, "XGBoost / ML Engine"),
        (GREEN_ACC, "Prediction Output"),
        (RED_ACC, "Validation (TCD)"),
        (PURPLE, "Clinician Interface"),
    ]
    for i, (clr, label) in enumerate(items):
        y_pos = legend_y + 25 - i * 22
        c.setFillColor(clr)
        c.roundRect(legend_x + 12, y_pos - 4, 12, 12, 2, fill=1, stroke=0)
        c.setFillColor(LIGHT_GRAY)
        c.drawString(legend_x + 30, y_pos - 1, label)
    c.restoreState()

    # Metrics badge
    badge_y = legend_y - 120
    _rounded_rect(c, legend_x + 5, badge_y, 155, 70, 6, DEEP_BLUE, ACCENT, 1)
    _draw_text_block(c, legend_x + 5, badge_y, 155, 70, [
        "Validation Metrics",
        "R² = 0.735",
        "MAE = 4.816%",
        "RMSE = 4.475%",
    ], 8.5, ACCENT)

    # Dataset badge
    ds_y = badge_y - 80
    _rounded_rect(c, legend_x + 5, ds_y, 155, 65, 6, DEEP_BLUE, TEAL, 1)
    _draw_text_block(c, legend_x + 5, ds_y, 155, 65, [
        "PhysioNet Dataset",
        "Cerebral Vasoregulation",
        "in Elderly with Stroke",
    ], 8, TEAL)

    # Footer
    c.saveState()
    c.setFillColor(HexColor("#555555"))
    c.setFont("Helvetica", 7)
    c.drawCentredString(W / 2, 15, "CerebroTwin © 2026 Harish Pannuru, Pavan Sai Krishna, Girish Chandra Kovuru  —  IIITDM Kancheepuram")
    c.restoreState()

    c.showPage()
    c.save()
    print(f"  ✓ Architecture diagram → {output_path}")


# ═══════════════════════════════════════════════════════════════════
# PART 2:  Presentation Deck  (multi-page landscape PDF)
# ═══════════════════════════════════════════════════════════════════

def _slide_background(c, W, H, slide_num, total, title, subtitle=None):
    """Draw consistent slide background, header, footer."""
    c.setFillColor(DARK_NAVY)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # Top accent bar
    c.setFillColor(ACCENT)
    c.rect(0, H - 6, W, 6, fill=1, stroke=0)

    # Title area
    _rounded_rect(c, 30, H - 75, W - 60, 55, 8, DEEP_BLUE, ACCENT, 1.5)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(W / 2, H - 56, title)
    if subtitle:
        c.setFont("Helvetica", 11)
        c.setFillColor(ACCENT)
        c.drawCentredString(W / 2, H - 72, subtitle)

    # Footer
    c.setFillColor(HexColor("#555555"))
    c.setFont("Helvetica", 7)
    c.drawString(30, 15, "CerebroTwin  |  IIITDM Kancheepuram  |  © 2026")
    c.drawRightString(W - 30, 15, f"Slide {slide_num} / {total}")

    # Bottom accent
    c.setFillColor(ACCENT)
    c.rect(0, 0, W, 3, fill=1, stroke=0)


def _bullet_list(c, x, y, items, font_size=11, line_spacing=20, color=LIGHT_GRAY, bullet_color=ACCENT):
    """Draw a bulleted list."""
    c.saveState()
    for i, item in enumerate(items):
        yp = y - i * line_spacing
        c.setFillColor(bullet_color)
        c.setFont("Helvetica-Bold", font_size)
        c.drawString(x, yp, "▸")
        c.setFillColor(color)
        c.setFont("Helvetica", font_size)
        c.drawString(x + 15, yp, item)
    c.restoreState()


def build_presentation_pdf(output_path: Path):
    """Build the multi-slide presentation deck."""
    W, H = landscape(LETTER)
    c = Canvas(str(output_path), pagesize=landscape(LETTER))
    c.setTitle("CerebroTwin – Project Presentation")
    TOTAL = 8

    # ── SLIDE 1: Title ──────────────────────────────────────────
    c.setFillColor(DARK_NAVY)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(ACCENT)
    c.rect(0, H - 6, W, 6, fill=1, stroke=0)
    c.rect(0, 0, W, 3, fill=1, stroke=0)

    # Large title
    _rounded_rect(c, 50, H - 200, W - 100, 150, 12, DEEP_BLUE, ACCENT, 2)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(W / 2, H - 95, "CerebroTwin")
    c.setFont("Helvetica", 13)
    c.setFillColor(ACCENT)
    c.drawCentredString(W / 2, H - 120, "A Personalized Digital Twin for Forecasting Orthostatic")
    c.drawCentredString(W / 2, H - 137, "Cerebral Blood-Flow-Velocity Instability in Type 2 Diabetes")

    c.setFillColor(LIGHT_GRAY)
    c.setFont("Helvetica", 14)
    c.drawCentredString(W / 2, H - 175, "Hackathon Submission — October 2026")

    # Team box
    team_y = H - 350
    _rounded_rect(c, 100, team_y, W - 200, 120, 10, NAVY, TEAL, 1.5)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(W / 2, team_y + 95, "Team Members")
    c.setFont("Helvetica", 11)
    members = [
        "Harish Pannuru  —  CSE  —  Lead ML & Full-Stack Architect",
        "Pavan Sai Krishna  —  MECH  —  Physiological Systems & Dynamics Lead",
        "Girish Chandra Kovuru  —  MECH  —  Biomedical Signal & Feature Engineer",
    ]
    for i, m in enumerate(members):
        c.setFillColor(LIGHT_GRAY)
        c.drawCentredString(W / 2, team_y + 70 - i * 22, m)

    # Institution
    c.setFillColor(TEAL)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(W / 2, team_y - 25, "Indian Institute of Information Technology, Design and Manufacturing, Kancheepuram")
    c.setFont("Helvetica", 10)
    c.setFillColor(LIGHT_GRAY)
    c.drawCentredString(W / 2, team_y - 42, "Chennai, Tamil Nadu, India")

    c.setFillColor(HexColor("#555555"))
    c.setFont("Helvetica", 7)
    c.drawString(30, 15, "CerebroTwin  |  IIITDM Kancheepuram  |  © 2026")
    c.drawRightString(W - 30, 15, f"Slide 1 / {TOTAL}")

    c.showPage()

    # ── SLIDE 2: Clinical Problem ─────────────────────────────
    _slide_background(c, W, H, 2, TOTAL,
                      "Clinical Problem Statement",
                      "Orthostatic Cerebral Hypoperfusion in Type 2 Diabetes")

    items = [
        "T2D patients develop autonomic neuropathy → impaired cerebral autoregulation",
        "Sit-to-stand posture change causes acute systemic blood pressure drop",
        "Healthy baroreflex compensates within seconds; T2D response is blunted & delayed",
        "Result: transient cerebral hypoperfusion → dizziness, syncope, falls, injury",
        "Brachial BP and HR alone cannot reflect cerebral hemodynamics",
        "TCD (Transcranial Doppler) measures MCAv directly — but is impractical for ambulatory use",
        "Unmet need: predict cerebral flow trajectory from non-cerebral bedside signals",
    ]
    _bullet_list(c, 60, H - 120, items, 11, 28)

    # Highlight box
    _rounded_rect(c, 60, 55, W - 120, 55, 8, DEEP_BLUE, ORANGE, 1.5)
    _draw_text_block(c, 60, 55, W - 120, 55, [
        "Key Insight: Systemic cardiovascular signals encode latent information",
        "about cerebral autoregulatory capacity that ML can decode",
    ], 11, ORANGE)

    c.showPage()

    # ── SLIDE 3: Healthcare Use Case ──────────────────────────
    _slide_background(c, W, H, 3, TOTAL,
                      "Healthcare Use Case & Research Scope",
                      "Non-Diagnostic Clinical Decision Support Framing")

    left_items = [
        "Target Population: Adults with Type 2 Diabetes (T2D)",
        "Clinical Setting: Pre-operative, geriatric, endocrine clinics",
        "Use Case: Forecast patient-specific MCAv trajectory",
        "Risk Stratification: Flag delayed cerebral recovery",
    ]
    _bullet_list(c, 60, H - 120, left_items, 10.5, 26)

    _rounded_rect(c, 60, H - 290, W - 120, 55, 8, HexColor("#3E1A1A"), RED_ACC, 1.5)
    _draw_text_block(c, 60, H - 290, W - 120, 55, [
        "⚠  Research Scope Disclaimer",
        "CerebroTwin is a proof-of-concept research tool — NOT a diagnostic device.",
        "Forecasts are flagged as deviations from control distributions, not disease labels.",
    ], 10, RED_ACC)

    scope_items = [
        "Outputs are decision-support signals, not prescriptive treatment recommendations",
        "Counterfactual simulations illustrate physiological sensitivities, not clinical advice",
        "All validation uses strictly held-out subjects (GroupKFold) — no data leakage",
    ]
    _bullet_list(c, 60, H - 320, scope_items, 10.5, 24)

    c.showPage()

    # ── SLIDE 4: Methodology ──────────────────────────────────
    _slide_background(c, W, H, 4, TOTAL,
                      "Digital Twin Methodology",
                      "PhysioNet Dataset Alignment & Physiological Feature Extraction")

    items = [
        "Data Source: PhysioNet 'Cerebral Vasoregulation in Elderly with Stroke' dataset",
        "Subjects: Healthy controls + stroke/T2D cohorts with concurrent multi-signal recordings",
        "Signals: Beat-to-beat arterial BP, ECG/HR, respiration, EtCO₂, and TCD MCAv",
        "Orthostatic Protocol: Supine → Sit → Stand with continuous 60-second monitoring",
    ]
    _bullet_list(c, 60, H - 120, items, 10.5, 26)

    # Feature categories
    _rounded_rect(c, 60, H - 330, 320, 150, 8, DEEP_BLUE, TEAL, 1.5)
    _draw_text_block(c, 60, H - 330, 320, 150, [
        "Non-Cerebral Features (Model Input)",
        "───────────────────────",
        "Static: Age, Sex, T2D, HbA1c, HTN, Meds, Resting HR/MAP",
        "HRV: SDNN, RMSSD, LF/HF ratio",
        "Dynamic: ΔMAP/Δt, ΔHR/Δt slopes (0–5s post-stand)",
        "Respiratory: Rate shifts, EtCO₂ baseline & drop",
    ], 9, TEAL)

    _rounded_rect(c, 410, H - 330, 320, 150, 8, HexColor("#3E1A1A"), RED_ACC, 1.5)
    _draw_text_block(c, 410, H - 330, 320, 150, [
        "Ground Truth (Validation Only)",
        "───────────────────────",
        "TCD Middle Cerebral Artery velocity (MCAv)",
        "Strictly withheld from model training inputs",
        "Used only as supervised learning target",
        "Subject-level isolation enforced",
    ], 9, RED_ACC)

    c.showPage()

    # ── SLIDE 5: ML Architecture ──────────────────────────────
    _slide_background(c, W, H, 5, TOTAL,
                      "Machine Learning Architecture",
                      "Strict Subject-Level GroupKFold Cross-Validation (K=5)")

    # Three model boxes
    models_info = [
        ("Model 0: Mean Baseline", [
            "Population mean control trajectory",
            "Reference benchmark for comparison",
        ], GREEN_ACC),
        ("Model 1: Ridge Regression", [
            "L2-regularized linear model",
            "Tests linear signal–cerebral relationship",
        ], ACCENT),
        ("Model 2: Quantile XGBoost", [
            "Multi-output gradient boosted trees",
            "Quantile regression (10th/50th/90th)",
            "90% prediction uncertainty intervals",
        ], ORANGE),
    ]
    for i, (title, descs, clr) in enumerate(models_info):
        bx = 50 + i * 245
        by = H - 280
        bw2 = 225
        bh2 = 145
        _rounded_rect(c, bx, by, bw2, bh2, 8, NAVY, clr, 2)
        c.setFillColor(clr)
        c.setFont("Helvetica-Bold", 11)
        c.drawCentredString(bx + bw2 / 2, by + bh2 - 20, title)
        for j, d in enumerate(descs):
            c.setFillColor(LIGHT_GRAY)
            c.setFont("Helvetica", 9.5)
            c.drawCentredString(bx + bw2 / 2, by + bh2 - 45 - j * 18, d)

    # GroupKFold highlight
    _rounded_rect(c, 60, 55, W - 120, 55, 8, DEEP_BLUE, PURPLE, 1.5)
    _draw_text_block(c, 60, 55, W - 120, 55, [
        "GroupKFold (K=5): Every subject appears in TRAIN or TEST, never both.",
        "Eliminates temporal & identity leakage — critical for medical ML validity.",
    ], 11, PURPLE)

    c.showPage()

    # ── SLIDE 6: Results ──────────────────────────────────────
    _slide_background(c, W, H, 6, TOTAL,
                      "Results & Validation Metrics",
                      "Quantile XGBoost on Held-Out Subject Folds")

    # Metrics boxes
    metrics = [
        ("R²", "0.735", "Coefficient of Determination", GREEN_ACC),
        ("MAE", "4.816%", "Mean Absolute Error (pct MCAv drop)", ACCENT),
        ("RMSE", "4.475%", "Trajectory RMSE (0–60s window)", ORANGE),
    ]
    for i, (name, val, desc, clr) in enumerate(metrics):
        mx = 60 + i * 240
        my = H - 230
        mw2 = 210
        mh2 = 115
        _rounded_rect(c, mx, my, mw2, mh2, 10, NAVY, clr, 2)
        c.setFillColor(clr)
        c.setFont("Helvetica-Bold", 32)
        c.drawCentredString(mx + mw2 / 2, my + mh2 - 45, val)
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(mx + mw2 / 2, my + mh2 - 70, name)
        c.setFillColor(LIGHT_GRAY)
        c.setFont("Helvetica", 8.5)
        c.drawCentredString(mx + mw2 / 2, my + 15, desc)

    validation_notes = [
        "All metrics computed on held-out test subjects (GroupKFold split)",
        "Model never sees TCD during training — pure non-cerebral → cerebral prediction",
        "Quantile heads provide 90% prediction intervals for clinical uncertainty quantification",
        "Recovery trajectory classified as Normal or Delayed based on 30-second threshold",
    ]
    _bullet_list(c, 60, H - 320, validation_notes, 10.5, 24)

    c.showPage()

    # ── SLIDE 7: SHAP Explainability ──────────────────────────
    _slide_background(c, W, H, 7, TOTAL,
                      "SHAP Explainability",
                      "Autonomic & Vascular Feature Drivers")

    items = [
        "SHAP (SHapley Additive exPlanations) via TreeExplainer for XGBoost",
        "Per-patient feature attribution — which inputs drive THIS patient's prediction?",
        "Top autonomic drivers: ΔMAP/Δt (BP drop slope), ΔHR/Δt (compensatory HR), LF/HF ratio",
        "Top vascular drivers: Resting MAP, baseline HRV (SDNN, RMSSD), EtCO₂ reactivity",
        "Clinical modifiers: HbA1c level, T2D duration, β-blocker status",
    ]
    _bullet_list(c, 60, H - 120, items, 11, 28)

    # Interpretability box
    _rounded_rect(c, 60, H - 380, W - 120, 80, 8, DEEP_BLUE, TEAL, 1.5)
    _draw_text_block(c, 60, H - 380, W - 120, 80, [
        "Clinical Interpretability",
        "Every prediction is accompanied by a SHAP waterfall plot showing",
        "exactly which physiological features pushed the forecast toward risk or safety.",
        "This transparency is essential for clinician trust in AI-assisted decision support.",
    ], 10.5, TEAL)

    c.showPage()

    # ── SLIDE 8: Counterfactual Simulator ─────────────────────
    _slide_background(c, W, H, 8, TOTAL,
                      "Counterfactual What-If Simulation",
                      "Clinician Decision Support Engine")

    items = [
        "Interactive \"What-If\" engine in the Streamlit Clinician Dashboard",
        "Modify any clinical input and instantly see updated MCAv trajectory forecast",
        "Example: \"What if HbA1c drops from 9.2% to 7.5% with medication?\"",
        "Example: \"What if a β-blocker is added to the current regimen?\"",
        "Side-by-side trajectory comparison: Baseline vs. Counterfactual scenario",
        "Helps clinicians explore physiological sensitivities before real interventions",
    ]
    _bullet_list(c, 60, H - 120, items, 11, 26)

    # Feature box
    _rounded_rect(c, 60, H - 400, W - 120, 100, 8, NAVY, PURPLE, 1.5)
    _draw_text_block(c, 60, H - 400, W - 120, 100, [
        "Simulator Capabilities",
        "──────────────────────────",
        "Adjust: Age, HbA1c, HTN status, Medications, Resting HR, MAP",
        "Visualize: Predicted trajectory shift, nadir change, recovery time delta",
        "Compare: Original vs. counterfactual with uncertainty bands overlay",
    ], 10, PURPLE)

    # Closing note
    _rounded_rect(c, 100, 50, W - 200, 50, 8, DEEP_BLUE, GREEN_ACC, 1.5)
    _draw_text_block(c, 100, 50, W - 200, 50, [
        "CerebroTwin — Bridging Systemic Signals to Cerebral Insight",
        "Thank you for reviewing our submission",
    ], 12, GREEN_ACC)

    c.showPage()
    c.save()
    print(f"  ✓ Presentation deck  → {output_path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    arch_path = DOCS_DIR / "CerebroTwin_Architecture_Diagram.pdf"
    pres_path = DOCS_DIR / "CerebroTwin_Project_Presentation.pdf"

    print("Generating CerebroTwin documentation PDFs …")
    build_architecture_pdf(arch_path)
    build_presentation_pdf(pres_path)

    # Verify files exist
    ok = True
    for p in [arch_path, pres_path]:
        if p.exists() and p.stat().st_size > 0:
            print(f"  ✓ Verified: {p}  ({p.stat().st_size:,} bytes)")
        else:
            print(f"  ✗ MISSING or EMPTY: {p}")
            ok = False

    if ok:
        print("\n✅ All documentation PDFs generated and verified successfully.")
    else:
        print("\n❌ Some PDFs failed to generate. Check errors above.")
        sys.exit(1)
