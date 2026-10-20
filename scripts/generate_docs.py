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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from scripts.generate_architecture import build_architecture_pdf


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

    print("Generating CerebroTwin documentation PDFs …")
    build_architecture_pdf(arch_path)

    # Check for user-provided presentation files in docs/
    user_pres_files = [
        p for p in DOCS_DIR.glob("*Presentation*.pdf")
        if p.name != "CerebroTwin_Project_Presentation.pdf"
    ]
    if user_pres_files:
        for p in user_pres_files:
            print(f"  [OK] Preserving existing manual presentation: {p.name} ({p.stat().st_size:,} bytes)")
    else:
        pres_path = DOCS_DIR / "CerebroTwin_Project_Presentation.pdf"
        if not pres_path.exists():
            build_presentation_pdf(pres_path)
        else:
            print(f"  [OK] Preserving existing presentation: {pres_path.name} ({pres_path.stat().st_size:,} bytes)")

    # Verify architecture diagram exists
    if arch_path.exists() and arch_path.stat().st_size > 0:
        print(f"  [OK] Verified: {arch_path.name}  ({arch_path.stat().st_size:,} bytes)")
        print("\nAll documentation verified successfully.")
    else:
        print(f"\n[FAIL] Architecture diagram failed to generate: {arch_path}")
        sys.exit(1)
