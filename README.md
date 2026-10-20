# CerebroTwin: A Personalized Digital Twin for Forecasting Orthostatic Cerebral Blood-Flow-Velocity Instability in Type 2 Diabetes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://github.com/harishpannuru/cerebrotwin)

---

## 👥 Team Details & Institutional Affiliation

### Institution
**Indian Institute of Information Technology, Design and Manufacturing, Kancheepuram (IIITDM Kancheepuram)**  
Chennai, Tamil Nadu, India

### Team Members & Role Distribution
Responsibilities were equally distributed across Machine Learning, Biomedical Dynamics, and Signal Engineering:

| Team Member | Department | Project Role | Core Contributions |
| :--- | :--- | :--- | :--- |
| **Harish Pannuru** | Computer Science & Engineering (CSE) | **Lead ML & Full-Stack Architect** | End-to-end ML pipeline, Multi-output Quantile XGBoost model, GroupKFold cross-validation, sys.path auto-bootstrap engine, and GitHub/Cloud deployment. |
| **Pavan Sai Krishna** | Mechanical Engineering (MECH) | **Physiological Systems & Dynamics Lead** | Cardiovascular & cerebrovascular coupling dynamics, orthostatic sit-to-stand transit modeling, baroreflex proxy formulations, and counterfactual simulation design. |
| **Girish Chandra Kovuru** | Mechanical Engineering (MECH) | **Biomedical Signal & Feature Engineer** | Beat-to-beat arterial blood pressure analysis, time/frequency domain HRV extraction, EtCO2 respiratory integration, and TCD ground-truth metric validation. |

---

## 🩺 Healthcare Use Case & Problem Statement

### Clinical Problem
Adults with Type 2 Diabetes (T2D) frequently develop autonomic neuropathy and impaired cerebral autoregulation. During sudden postural changes (such as transitioning from sitting to standing), systemic blood pressure drops. While healthy individuals compensate rapidly via autonomic baroreflex and cerebrovascular vasodilation, patients with T2D often experience blunt, sluggish compensatory responses, leading to acute cerebral hypoperfusion, orthostatic dizziness, syncope, and heightened fall risk.

Current clinical monitoring relies primarily on systemic physiological signals (brachial blood pressure, heart rate), which fail to reflect cerebral hemodynamics. Transcranial Doppler (TCD) ultrasonography can measure middle cerebral artery blood-flow velocity (MCAv), but continuous ambulatory TCD monitoring is clinically impractical and technically cumbersome.

### The CerebroTwin Solution
**CerebroTwin** is a personalized physiological Digital Twin that predicts an individual's 0–60 second middle cerebral artery blood-flow-velocity (MCAv) response during a sit-to-stand orthostatic challenge using only non-cerebral cardiovascular and respiratory inputs coupled with their baseline clinical profile.

> **Research Scope & Intended Use Disclaimer:**  
> CerebroTwin is a proof-of-concept research and clinical decision-support system. Forecast abnormalities are flagged as a research-oriented deviation relative to control distributions rather than a diagnostic disease label. It does not provide medical prescriptions or clinical treatment instructions.

---

## 🏗️ Technical Stack

- **Core Language:** Python 3.11
- **Machine Learning & Modeling:** XGBoost, LightGBM, Scikit-Learn (Ridge Regularization), NumPy, SciPy
- **Explainable AI (XAI):** SHAP (SHapley Additive exPlanations - TreeExplainer)
- **Signal Processing & Biometrics:** SciPy Signal, dynamic trapezoidal integration (`np.trapezoid` / `np.trapz`), WFDB (Waveform Database)
- **User Interface & Scientific Visualization:** Streamlit, Plotly Graph Objects
- **Model Persistence & Deployment:** Joblib, Render Blueprint (`render.yaml`), Streamlit Community Cloud, Cloudflare Quick Tunnels

---

## 🧠 AI / ML Model Architecture & Validation

### 1. Physiological Feature Pipeline (Non-Cerebral Inputs Only)
*Runtime inputs strictly exclude TCD signals:*
- **Static Clinical Baseline:** Age, sex, T2D status, HbA1c, hypertension status, medications, resting HR, resting MAP, baseline HRV (SDNN, RMSSD, LF/HF), baseline respiration.
- **Dynamic Transition Signals:** 0–5s post-stand arterial blood pressure drop slope (ΔMAP/Δt), heart rate compensatory acceleration slope (ΔHR/Δt), respiratory rate shifts, and end-tidal CO₂ (EtCO₂).

### 2. Model Progression
1. **Model 0 (Baseline):** Mean control response baseline.
2. **Model 1 (Regularized Linear):** Ridge Regression with L2 regularization.
3. **Model 2 (Production Digital Twin):** Multi-output Gradient Boosted Decision Trees (`XGBoost`) with quantile regression heads providing 90% prediction intervals (±uncertainty).

### 3. Primary Target Outputs (t = 0 to 60 s)
- Predicted percentage reduction in MCAv from baseline (`pct_mca_drop`)
- Predicted minimum velocity / Nadir (MCAv_nadir in cm/s) and time-to-nadir
- Predicted cumulative cerebral-flow-velocity deficit (AUC below baseline)
- Predicted recovery trajectory and classification (`Normal` vs. `Delayed`)
- Research response status (`NORMAL` vs. `HIGH DEVIATION` relative to reference control cohort)

### 4. Zero-Leakage GroupKFold Cross-Validation
- Model evaluation strictly enforces **Subject-Level GroupKFold cross-validation (K=5)**.
- Any given subject ID (e.g., `S0250`) exists **strictly in train OR test, NEVER both**, eliminating temporal and identity data leakage.
- **Validation Metrics Achieved:**
  - R² = 0.735 on primary peak drop
  - MAE = 4.816% on percentage MCAv reduction
  - Trajectory RMSE = 4.475% over the continuous 0–60 second response window

---

## 📊 Architecture Diagram & Project Presentation

Both official project documents are located in the [`/docs`](./docs) directory and are fully publicly accessible without special permissions:

- 📄 **Architecture Diagram (PDF):** [`docs/CerebroTwin_Architecture_Diagram.pdf`](./docs/CerebroTwin_Architecture_Diagram.pdf)
- 📊 **Project Presentation Deck (PDF):** [`docs/CerebroTwin_Project_Presentation.pdf`](./docs/CerebroTwin_Project_Presentation.pdf)

---

## 🎥 Demonstration Video (15–20 Minutes)

- **Video Link:** `[DEMO VIDEO LINK - UNLISTED YOUTUBE URL PENDING UPLOAD]`
- *(A complete 15–20 minute walkthrough covering problem motivation, Digital Twin architecture, GroupKFold validation, live Streamlit patient replay, SHAP explainability, and counterfactual simulation will be linked here upon final render).*

---

## 🚀 Quickstart & Reproduction

```bash
# Clone the repository
git clone https://github.com/harishpannuru/cerebrotwin.git
cd cerebrotwin

# Install dependencies (Python 3.11 recommended)
pip install -r requirements.txt

# Run full pipeline (Auto-bootstrapped if artifacts are already present)
python main.py train
python main.py evaluate

# Launch the Clinician Dashboard
streamlit run src/app/dashboard.py
```

---

## 📁 Repository Structure

```
cerebrotwin/
├── LICENSE                          # MIT License
├── README.md                        # This file
├── .python-version                  # Python 3.11 runtime pin
├── requirements.txt                 # Python dependencies
├── config.yaml                      # Pipeline configuration
├── main.py                          # CLI entry point (train / evaluate)
├── render.yaml                      # Render deployment blueprint
├── test_pipeline.py                 # End-to-end pipeline tests
├── docs/
│   ├── CerebroTwin_Architecture_Diagram.pdf
│   └── CerebroTwin_Project_Presentation.pdf
├── scripts/
│   ├── download_physionet_subset.py # PhysioNet data downloader
│   └── generate_docs.py            # PDF generation script
├── data/
│   └── synthetic/                   # Synthetic training data
│       ├── metadata.csv
│       └── waveforms/
├── models/                          # Trained model artifacts
│   ├── xgboost_twin.joblib
│   ├── ridge_baseline.joblib
│   ├── mean_baseline.joblib
│   ├── feature_names.joblib
│   ├── target_names.joblib
│   ├── eval_dict.joblib             # Cached GroupKFold evaluation results
│   └── evaluation_results.csv
└── src/
    ├── __init__.py
    ├── app/
    │   ├── __init__.py
    │   └── dashboard.py             # Streamlit Clinician Dashboard
    ├── data/
    │   ├── __init__.py
    │   ├── loader.py
    │   ├── preprocessing.py
    │   └── synthetic_generator.py
    ├── features/
    │   ├── __init__.py
    │   ├── digital_twin_state.py
    │   └── feature_engineering.py
    └── models/
        ├── __init__.py
        ├── baselines.py
        ├── counterfactuals.py
        ├── evaluator.py
        └── main_xgboost.py
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [`LICENSE`](./LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PhysioNet** for providing the open-access physiological signal datasets used in model development and validation.
- **IIITDM Kancheepuram** for institutional support and computational resources.
- **SHAP** (Lundberg & Lee) for the interpretable machine learning framework enabling clinical transparency.

---

*Built with ❤️ for advancing cerebrovascular health monitoring in diabetes care.*
