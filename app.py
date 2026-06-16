import sys
if sys.version_info >= (3, 13):
    import types
    sys.modules['audioop'] = types.ModuleType('audioop')

import os
import gradio as gr
import pandas as pd
import numpy as np
from inference import predict_churn, get_pipeline

try:
    _meta      = get_pipeline()["metadata"]
    MODEL_VER  = _meta.get("model_version", "2.0.0")
    PR_AUC     = _meta["performance"]["PR_AUC"]
    ROC_AUC    = _meta["performance"]["ROC_AUC"]
    RECALL     = _meta["performance"]["Recall"]
    CAPTURE_20 = _meta["performance"]["Capture_rate_top20pct"]
    CV_MEAN    = _meta["cv_performance"]["PR_AUC_mean"]
    CV_STD     = _meta["cv_performance"]["PR_AUC_std"]
    THRESHOLD  = _meta["decision_threshold"]
    N_FEAT     = _meta["n_features"]
except Exception:
    MODEL_VER  = "2.0.0"
    PR_AUC = ROC_AUC = RECALL = CAPTURE_20 = CV_MEAN = CV_STD = THRESHOLD = 0.0
    N_FEAT = 46

# ── Styling ───────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

body, .gradio-container {
    background: #0f172a !important;
    min-height: 100vh;
}
.gradio-container {
    max-width: 1300px !important;
    margin: 0 auto !important;
    padding: 16px !important;
}

/* Labels */
label, .gr-form label {
    color: #cbd5e1 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}

/* Inputs */
input[type=number], .gr-number input,
select, .gr-dropdown select,
textarea, .gr-textbox textarea {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s !important;
}
input:focus, select:focus {
    border-color: #6366f1 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}

/* Group panels */
.gr-group, .gr-box {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

/* Checkboxes */
input[type=checkbox] { accent-color: #6366f1 !important; }

/* Predict button */
#predict-btn button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: 14px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.25s !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
    letter-spacing: 0.3px !important;
}
#predict-btn button:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 28px rgba(99,102,241,0.5) !important;
}

/* Example buttons */
#ex-h button { background: rgba(239,68,68,0.12) !important; border: 1px solid rgba(239,68,68,0.4) !important; color: #fca5a5 !important; border-radius:8px !important; font-weight:600 !important; transition:all 0.2s !important; }
#ex-m button { background: rgba(245,158,11,0.12) !important; border: 1px solid rgba(245,158,11,0.4) !important; color: #fcd34d !important; border-radius:8px !important; font-weight:600 !important; transition:all 0.2s !important; }
#ex-l button { background: rgba(16,185,129,0.12) !important; border: 1px solid rgba(16,185,129,0.4) !important; color: #6ee7b7 !important; border-radius:8px !important; font-weight:600 !important; transition:all 0.2s !important; }
#ex-h button:hover { background: rgba(239,68,68,0.25) !important; }
#ex-m button:hover { background: rgba(245,158,11,0.25) !important; }
#ex-l button:hover { background: rgba(16,185,129,0.25) !important; }

/* Output textboxes */
.gr-textbox textarea, .output-text textarea {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    text-align: center !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
"""

# ── Static HTML blocks ────────────────────────────────────────────
HEADER = """
<div style="background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 50%,#06b6d4 100%);
            border-radius:14px;padding:32px 40px;margin-bottom:18px;text-align:center;
            box-shadow:0 20px 60px rgba(99,102,241,0.3);">
  <div style="font-size:2.1rem;font-weight:800;color:#fff;letter-spacing:-0.5px;">
    📡 Telecom Customer Churn Prediction
  </div>
  <div style="color:rgba(255,255,255,0.88);margin:8px 0 0;font-size:0.95rem;font-weight:400;">
    AI-powered retention intelligence &nbsp;·&nbsp; 1,000,000 records trained
  </div>
  <div style="display:inline-block;background:rgba(255,255,255,0.18);
              border:1px solid rgba(255,255,255,0.3);color:#fff;
              padding:5px 16px;border-radius:20px;font-size:0.73rem;
              font-weight:700;margin-top:12px;letter-spacing:0.6px;text-transform:uppercase;">
    CatBoost + Isotonic Calibration &nbsp;|&nbsp; %(nf)d features &nbsp;|&nbsp; v%(ver)s
  </div>
</div>
""" % {"nf": N_FEAT, "ver": MODEL_VER}

METRICS = """
<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:18px;">
""" + "".join([
    f"""<div style="background:#1e293b;border:1px solid #334155;border-radius:12px;
                   padding:16px 10px;text-align:center;">
          <div style="font-size:1.45rem;font-weight:800;color:#6366f1;">{v}</div>
          <div style="font-size:0.68rem;color:#64748b;margin-top:5px;font-weight:600;
                      text-transform:uppercase;letter-spacing:0.6px;">{l}</div>
        </div>"""
    for v, l in [
        (f"{ROC_AUC:.3f}", "ROC-AUC"),
        (f"{PR_AUC:.3f}",  "PR-AUC"),
        (f"{RECALL:.0%}",  "Recall"),
        (f"{CV_MEAN:.3f}±{CV_STD:.3f}", "5-Fold CV"),
        (f"{CAPTURE_20:.0%}", "Capture @Top 20%"),
    ]
]) + "</div>"

PLACEHOLDER = """
<div style="background:#1e293b;border:1px solid #334155;border-radius:14px;
            padding:60px 30px;text-align:center;min-height:540px;
            display:flex;flex-direction:column;align-items:center;justify-content:center;">
  <div style="font-size:3.5rem;margin-bottom:16px;">📡</div>
  <div style="font-size:1.25rem;font-weight:700;color:#f1f5f9;margin-bottom:10px;">
    Ready for Prediction
  </div>
  <div style="color:#64748b;font-size:0.88rem;max-width:300px;line-height:1.65;">
    Fill in the customer profile on the left, then click
    <span style="color:#6366f1;font-weight:600;">Predict Churn</span>
    to see the risk analysis.
  </div>
  <div style="margin-top:24px;padding:12px 20px;background:#0f172a;
              border-radius:8px;border:1px solid #1e293b;">
    <span style="color:#475569;font-size:0.78rem;">
      Trained on <strong style="color:#6366f1;">1,000,000</strong> customers
      &nbsp;·&nbsp; 46 features
      &nbsp;·&nbsp; CV stable ✅
    </span>
  </div>
</div>
"""

SHAP_DRIVERS = """
<div style="background:#1e293b;border:1px solid #334155;border-radius:14px;padding:20px;margin-top:14px;">
  <div style="font-size:0.72rem;font-weight:700;color:#6366f1;text-transform:uppercase;
              letter-spacing:1px;margin-bottom:14px;padding-bottom:10px;
              border-bottom:1px solid #334155;">
    Top Churn Drivers — SHAP Analysis
  </div>
  <div style="color:#475569;font-size:0.75rem;margin-bottom:14px;">
    Mean |SHAP| from 1M-record training run
  </div>
""" + "".join([
    f"""<div style="display:flex;align-items:center;gap:10px;padding:5px 0;">
          <div style="width:140px;font-size:0.75rem;color:#94a3b8;text-align:right;flex-shrink:0;">{nm}</div>
          <div style="flex:1;height:7px;background:#0f172a;border-radius:4px;overflow:hidden;">
            <div style="width:{int(sv/0.4379*100)}%;height:100%;background:{cl};border-radius:4px;"></div>
          </div>
          <div style="width:38px;font-size:0.72rem;color:#f1f5f9;font-weight:700;">{sv:.3f}</div>
        </div>"""
    for nm, sv, cl in [
        ("Contract Type",      0.4379, "#6366f1"),
        ("Payment Risk",       0.1942, "#8b5cf6"),
        ("Satisfaction Score", 0.1524, "#06b6d4"),
        ("Online Security",    0.1013, "#10b981"),
        ("Tech Support",       0.0987, "#f59e0b"),
        ("Svc Vulnerability",  0.0572, "#ef4444"),
        ("Contract+Sat Risk",  0.0276, "#f97316"),
        ("Complaint Trend",    0.0113, "#84cc16"),
    ]
]) + "</div>"

PERF_CARD = f"""
<div style="background:#1e293b;border:1px solid #334155;border-radius:14px;padding:20px;margin-top:14px;">
  <div style="font-size:0.72rem;font-weight:700;color:#6366f1;text-transform:uppercase;
              letter-spacing:1px;margin-bottom:14px;padding-bottom:10px;
              border-bottom:1px solid #334155;">
    Model Performance
  </div>
""" + "".join([
    f"""<div style="display:flex;justify-content:space-between;padding:8px 0;
                   border-bottom:1px solid #0f172a;">
          <span style="color:#64748b;font-size:0.8rem;">{k}</span>
          <span style="color:{c};font-size:0.8rem;font-weight:700;">{v}</span>
        </div>"""
    for k, v, c in [
        ("ROC-AUC",       f"{ROC_AUC:.4f}",            "#6366f1"),
        ("PR-AUC",        f"{PR_AUC:.4f}",              "#6366f1"),
        ("5-Fold CV",     f"{CV_MEAN:.4f} ± {CV_STD:.4f}", "#10b981"),
        ("Recall",        f"{RECALL:.1%}",              "#f59e0b"),
        ("Capture@Top20%",f"{CAPTURE_20:.1%}",          "#06b6d4"),
        ("Threshold",     str(THRESHOLD),               "#94a3b8"),
    ]
]) + "</div>"

FOOTER = f"""
<div style="text-align:center;padding:20px;color:#334155;font-size:0.78rem;
            margin-top:20px;border-top:1px solid #1e293b;">
  Built by
  <a href="https://huggingface.co/Simosro" style="color:#6366f1;text-decoration:none;">Simosro</a>
  &nbsp;·&nbsp; CatBoost v{MODEL_VER} + Isotonic Calibration
  &nbsp;·&nbsp; Trained on 1M telecom records
  &nbsp;·&nbsp;
  <a href="https://www.kaggle.com/azmainhaq" style="color:#6366f1;text-decoration:none;">Kaggle</a>
</div>
"""


# ── Result builder ────────────────────────────────────────────────
def build_result(res):
    prob   = res["churn_probability"]
    pred   = res["churn_predicted"]
    risk   = res["risk_tier"]
    value  = res["value_tier"]
    seg    = res["segment"]
    action = res["recommended_action"]
    pct    = prob * 100

    if pct >= 60:
        pc = "#ef4444"; bc = "linear-gradient(90deg,#ef4444,#dc2626)"
    elif pct >= 35:
        pc = "#f59e0b"; bc = "linear-gradient(90deg,#f59e0b,#d97706)"
    else:
        pc = "#10b981"; bc = "linear-gradient(90deg,#10b981,#059669)"

    vtext = "⚠️  LIKELY TO CHURN" if pred == 1 else "✅  LIKELY TO RETAIN"
    vbg   = "rgba(239,68,68,0.12)"  if pred == 1 else "rgba(16,185,129,0.12)"
    vco   = "#ef4444" if pred == 1 else "#10b981"
    vbo   = "rgba(239,68,68,0.35)"  if pred == 1 else "rgba(16,185,129,0.35)"

    seg_k = seg.split(":")[0].strip() if ":" in seg else "D"
    act_styles = {
        "A": ("rgba(239,68,68,0.1)",  "#ef4444", "#fca5a5"),
        "B": ("rgba(16,185,129,0.1)", "#10b981", "#6ee7b7"),
        "C": ("rgba(245,158,11,0.1)", "#f59e0b", "#fcd34d"),
        "D": ("rgba(99,102,241,0.1)", "#6366f1", "#a5b4fc"),
    }
    abg, abo, atx = act_styles.get(seg_k, act_styles["D"])

    rows = "".join([
        f"""<div style="display:flex;justify-content:space-between;padding:9px 0;
                       border-bottom:1px solid #1e293b;">
              <span style="color:#64748b;font-size:0.82rem;">{k}</span>
              <span style="color:#f1f5f9;font-size:0.82rem;font-weight:600;">{v}</span>
            </div>"""
        for k, v in [("Risk Tier", risk), ("Value Tier", value), ("Segment", seg)]
    ])

    return f"""
<div style="background:#1e293b;border:1px solid #334155;border-radius:14px;padding:24px;">
  <div style="font-size:0.72rem;font-weight:700;color:#6366f1;text-transform:uppercase;
              letter-spacing:1px;margin-bottom:18px;padding-bottom:10px;
              border-bottom:1px solid #334155;">
    Churn Prediction Result
  </div>

  <div style="font-size:3.8rem;font-weight:800;text-align:center;color:{pc};line-height:1;
              margin-bottom:6px;">{pct:.1f}%</div>
  <div style="text-align:center;color:#475569;font-size:0.78rem;margin-bottom:14px;">
    Churn Probability
  </div>

  <div style="margin-bottom:14px;">
    <div style="height:10px;background:#0f172a;border-radius:5px;overflow:hidden;">
      <div style="width:{min(pct,100):.0f}%;height:100%;background:{bc};
                  border-radius:5px;transition:width 0.6s ease;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;
                margin-top:5px;font-size:0.68rem;color:#334155;">
      <span>0%</span><span>Low</span><span>Medium</span><span>High</span><span>100%</span>
    </div>
  </div>

  <div style="text-align:center;font-size:0.98rem;font-weight:700;padding:10px 20px;
              border-radius:24px;background:{vbg};color:{vco};
              border:1px solid {vbo};margin-bottom:18px;">{vtext}</div>

  {rows}

  <div style="background:{abg};border-left:4px solid {abo};border-radius:0 8px 8px 0;
              padding:12px 16px;margin-top:16px;color:{atx};
              font-size:0.83rem;line-height:1.55;">
    <strong style="display:block;margin-bottom:4px;">💡 Recommended Action</strong>
    {action}
  </div>
</div>
"""


# ── Prediction logic ──────────────────────────────────────────────
def predict(
    age, gender, income, edu, marital, deps,
    tenure, contract, payment, paperless, senior,
    monthly, total, num_svc,
    has_ph, has_int, has_sec, has_bkp,
    has_dev, has_tec, has_stv, has_smv,
    sat, compl, svc_calls, late_pay, avg_gb, days_last, credit
):
    try:
        customer = {
            "age": int(age), "gender": str(gender),
            "annual_income": float(income), "education": str(edu),
            "marital_status": str(marital), "dependents": int(deps),
            "tenure": int(tenure), "contract": str(contract),
            "payment_method": str(payment),
            "paperless_billing": str(paperless),
            "senior_citizen": int(senior),
            "monthlycharges": float(monthly),
            "totalcharges": float(total),
            "num_services": int(num_svc),
            "has_phone_service": int(has_ph),
            "has_internet_service": int(has_int),
            "has_online_security": int(has_sec),
            "has_online_backup": int(has_bkp),
            "has_device_protection": int(has_dev),
            "has_tech_support": int(has_tec),
            "has_streaming_tv": int(has_stv),
            "has_streaming_movies": int(has_smv),
            "customer_satisfaction": float(sat),
            "num_complaints": float(compl),
            "num_service_calls": int(svc_calls),
            "late_payments": int(late_pay),
            "avg_monthly_gb": float(avg_gb),
            "days_since_last_interaction": int(days_last),
            "credit_score": float(credit),
        }
        return build_result(predict_churn(customer))
    except Exception as e:
        return (
            f"<div style='background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);"
            f"border-radius:12px;padding:32px;text-align:center;color:#fca5a5;'>"
            f"<div style='font-size:2rem;'>⚠️</div>"
            f"<div style='font-weight:700;margin:8px 0;'>Prediction Error</div>"
            f"<div style='font-size:0.83rem;opacity:0.8;'>{e}</div></div>"
        )


# ── Example presets ───────────────────────────────────────────────
HIGH_RISK = [45,"Male",45000,"Bachelor","Single",0,3,"month_to_month",
             "Electronic check","Yes",0,89.5,268.5,2,1,1,0,0,0,0,0,0,2,3,5,2,12.0,75,520.0]
LOW_RISK  = [38,"Female",72000,"Master","Married",2,54,"two_year",
             "Bank transfer (automatic)","No",0,105.2,5680.8,6,1,1,1,1,1,1,1,1,9,0,1,0,45.0,7,790.0]
MID_RISK  = [52,"Male",58000,"Bachelor","Divorced",1,18,"one_year",
             "Credit card (automatic)","Yes",0,72.3,1301.4,4,1,1,0,1,0,1,1,0,5,1,2,1,28.0,35,640.0]


# ── Build UI ──────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="Telecom Churn Prediction") as demo:

    gr.HTML(HEADER)
    gr.HTML(METRICS)

    with gr.Row():
        # ── LEFT: Input form ─────────────────────────────────────
        with gr.Column(scale=5):

            with gr.Group():
                gr.HTML('<div style="font-size:0.72rem;font-weight:700;color:#6366f1;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">👤 Demographics</div>')
                with gr.Row():
                    age    = gr.Number(label="Age", value=35, minimum=18, maximum=90)
                    gender = gr.Dropdown(label="Gender", choices=["Male","Female","Other"], value="Male")
                    senior = gr.Radio(label="Senior Citizen",
                               choices=[(">= 65 yrs", 1), ("< 65 yrs", 0)], value=0)
                with gr.Row():
                    income = gr.Number(label="Annual Income ($)", value=55000)
                    edu    = gr.Dropdown(label="Education",
                               choices=["Bachelor","Master","High School","PhD","Associate"],
                               value="Bachelor")
                with gr.Row():
                    marital = gr.Dropdown(label="Marital Status",
                                choices=["Single","Married","Divorced","Widowed"], value="Single")
                    deps    = gr.Slider(label="Dependents", minimum=0, maximum=5, step=1, value=0)

            with gr.Group():
                gr.HTML('<div style="font-size:0.72rem;font-weight:700;color:#6366f1;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;margin-top:4px;">📋 Account & Contract</div>')
                with gr.Row():
                    tenure   = gr.Slider(label="Tenure (months)", minimum=1, maximum=72, step=1, value=12)
                    contract = gr.Dropdown(label="Contract Type",
                                 choices=["month_to_month","one_year","two_year"],
                                 value="month_to_month")
                with gr.Row():
                    payment   = gr.Dropdown(label="Payment Method",
                                  choices=["Electronic check","Mailed check",
                                           "Bank transfer (automatic)","Credit card (automatic)"],
                                  value="Electronic check")
                    paperless = gr.Radio(label="Paperless Billing", choices=["Yes","No"], value="Yes")

            with gr.Group():
                gr.HTML('<div style="font-size:0.72rem;font-weight:700;color:#6366f1;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;margin-top:4px;">💳 Billing & Usage</div>')
                with gr.Row():
                    monthly = gr.Number(label="Monthly Charges ($)", value=65.0)
                    total   = gr.Number(label="Total Charges ($)",   value=780.0)
                    avg_gb  = gr.Number(label="Avg Monthly GB",      value=20.0)
                with gr.Row():
                    num_svc = gr.Slider(label="Active Services",  minimum=1, maximum=8,   step=1, value=3)
                    credit  = gr.Slider(label="Credit Score",     minimum=300, maximum=850, step=5, value=650)

            with gr.Group():
                gr.HTML('<div style="font-size:0.72rem;font-weight:700;color:#6366f1;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;margin-top:4px;">📡 Subscribed Services</div>')
                with gr.Row():
                    has_ph  = gr.Checkbox(label="Phone Service",    value=True)
                    has_int = gr.Checkbox(label="Internet Service", value=True)
                    has_sec = gr.Checkbox(label="Online Security",  value=False)
                    has_bkp = gr.Checkbox(label="Online Backup",    value=False)
                with gr.Row():
                    has_dev = gr.Checkbox(label="Device Protection", value=False)
                    has_tec = gr.Checkbox(label="Tech Support",      value=False)
                    has_stv = gr.Checkbox(label="Streaming TV",      value=False)
                    has_smv = gr.Checkbox(label="Streaming Movies",  value=False)

            with gr.Group():
                gr.HTML('<div style="font-size:0.72rem;font-weight:700;color:#6366f1;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;margin-top:4px;">📊 Behavior & Risk Signals</div>')
                with gr.Row():
                    sat   = gr.Slider(label="Satisfaction (1-10)", minimum=1, maximum=10, step=1, value=6)
                    compl = gr.Slider(label="Complaints (last year)", minimum=0, maximum=8, step=1, value=0)
                with gr.Row():
                    svc_calls = gr.Slider(label="Service Calls (last month)", minimum=0, maximum=12, step=1, value=1)
                    late_pay  = gr.Slider(label="Late Payments (last 3mo)",   minimum=0, maximum=5,  step=1, value=0)
                    days_last = gr.Slider(label="Days Since Last Contact",     minimum=1, maximum=365, step=1, value=30)

            gr.HTML('<div style="font-size:0.72rem;font-weight:700;color:#6366f1;text-transform:uppercase;letter-spacing:1px;margin:16px 0 10px;">⚡ Quick Load Examples</div>')
            with gr.Row():
                ex_h = gr.Button("🔴 High Risk",   elem_id="ex-h", size="sm")
                ex_m = gr.Button("🟡 Medium Risk", elem_id="ex-m", size="sm")
                ex_l = gr.Button("🟢 Low Risk",    elem_id="ex-l", size="sm")

            gr.HTML('<div style="margin:14px 0;"></div>')
            pred_btn = gr.Button("🔮  Predict Churn", elem_id="predict-btn", variant="primary")

        # ── RIGHT: Output ────────────────────────────────────────
        with gr.Column(scale=5):
            out = gr.HTML(value=PLACEHOLDER)
            gr.HTML(SHAP_DRIVERS)
            gr.HTML(PERF_CARD)

    gr.HTML(FOOTER)

    all_inputs = [
        age, gender, income, edu, marital, deps,
        tenure, contract, payment, paperless, senior,
        monthly, total, num_svc,
        has_ph, has_int, has_sec, has_bkp,
        has_dev, has_tec, has_stv, has_smv,
        sat, compl, svc_calls, late_pay, avg_gb, days_last, credit
    ]

    pred_btn.click(fn=predict, inputs=all_inputs, outputs=out)
    ex_h.click(fn=lambda: HIGH_RISK, inputs=[], outputs=all_inputs)
    ex_m.click(fn=lambda: MID_RISK,  inputs=[], outputs=all_inputs)
    ex_l.click(fn=lambda: LOW_RISK,  inputs=[], outputs=all_inputs)

demo.launch(server_name="0.0.0.0", server_port=7860)
