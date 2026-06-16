"""
inference.py — Telecom Churn Prediction
Production inference module for HuggingFace Space.
Loads all saved artifacts and exposes predict_churn().
"""

import os, json, pickle
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_artifacts")

# ── Global pipeline (loaded once at startup) ─────────────────────
_PIPELINE = None

def load_pipeline(model_dir: str = MODEL_DIR) -> dict:
    pipeline = {}
    with open(f"{model_dir}/pipeline_metadata.json") as f:
        pipeline["metadata"] = json.load(f)
    with open(f"{model_dir}/calibrated_model.pkl", "rb") as f:
        pipeline["model"] = pickle.load(f)
    with open(f"{model_dir}/label_encoders.pkl", "rb") as f:
        pipeline["encoders"] = pickle.load(f)
    with open(f"{model_dir}/fe_constants.json") as f:
        pipeline["fe_constants"] = json.load(f)
    return pipeline


def get_pipeline() -> dict:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = load_pipeline()
    return _PIPELINE


# ── Feature engineering (exact replica of training) ──────────────
def engineer_features(df: pd.DataFrame, fe: dict) -> pd.DataFrame:
    df = df.copy()
    svc_cols = [c for c in [
        "has_phone_service", "has_internet_service",
        "has_online_security", "has_online_backup",
        "has_device_protection", "has_tech_support",
        "has_streaming_tv", "has_streaming_movies"
    ] if c in df.columns]

    # Original 14 features
    df["charge_per_service"]     = df["monthlycharges"] / df["num_services"].clip(lower=1)
    df["charge_ratio"]           = df["totalcharges"] / (df["monthlycharges"] * df["tenure"] + 1)
    df["complaint_rate"]         = df["num_complaints"] / (df["tenure"] + 1)
    df["service_call_intensity"] = df["num_service_calls"] / (df["tenure"] + 1)
    df["payment_risk_score"]     = (df["late_payments"] * 3 +
                                    df["num_complaints"] * 2 +
                                    df["num_service_calls"] * 1)
    df["is_new_customer"]        = (df["tenure"] < fe["new_customer_months"]).astype(int)
    df["tenure_tier"]            = pd.cut(
        df["tenure"],
        bins=fe["tenure_bins"],
        labels=fe["tenure_labels"]
    ).astype(float).fillna(4).astype(int)
    df["recency_risk"]           = (df["days_since_last_interaction"] > fe["recency_risk_days"]).astype(int)
    df["satisfaction_risk"]      = (df["customer_satisfaction"] <= fe["satisfaction_risk_max"]).astype(int)
    df["interaction_gap_score"]  = np.log1p(df["days_since_last_interaction"])
    df["bundle_score"]           = df[svc_cols].sum(axis=1) if svc_cols else 0
    df["has_full_bundle"]        = (df["bundle_score"] >= fe["full_bundle_min"]).astype(int)

    if "contract" in df.columns:
        is_monthly = df["contract"].astype(str).str.lower().str.contains("month").astype(int)
        df["new_monthly_risk"]  = df["is_new_customer"] * is_monthly
        df["contract_sat_risk"] = is_monthly * (10 - df["customer_satisfaction"])

    if "credit_score" in df.columns:
        df["credit_risk_flag"] = (df["credit_score"] < fe["credit_risk_threshold"]).astype(int)

    # 3 confirmed new features
    df["complaint_trend"] = df["num_complaints"] / (df["tenure"] ** 2 + 1)

    if "has_tech_support" in df.columns:
        df["service_vulnerability"] = (1 - df["has_tech_support"]) * df["num_service_calls"]

    return df


def encode_features(df: pd.DataFrame, encoders: dict, feature_list: list) -> pd.DataFrame:
    df = df.copy()
    for col, enc in encoders.items():
        if col in df.columns:
            known = set(enc.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known else enc.classes_[0]
            )
            df[col] = enc.transform(df[col])
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0
    return df[feature_list]


# ── Segment + action mapping ──────────────────────────────────────
def get_segment(prob: float, monthly_charges: float,
                tiers: dict, median_charge: float) -> tuple:
    if prob >= tiers["high_risk_min"]:
        risk = "High Risk"
    elif prob >= tiers["low_risk_max"]:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    value = "High Value" if monthly_charges >= median_charge else "Low Value"

    mapping = {
        ("High Value", "High Risk"):   ("A: PRIORITY SAVE",
                                        "Immediate outreach — offer contract upgrade + 20% discount"),
        ("High Value", "Medium Risk"):  ("B: LOYALTY REWARD",
                                        "Enroll in loyalty program — proactive low-cost retention"),
        ("High Value", "Low Risk"):     ("B: LOYALTY REWARD",
                                        "Stable high-value customer — reward loyalty"),
        ("Low Value",  "High Risk"):    ("C: LET GO / MONITOR",
                                        "Monitor only — intervention cost may exceed revenue"),
        ("Low Value",  "Medium Risk"):  ("D: UPSELL TARGET",
                                        "Pitch additional services to grow customer value"),
        ("Low Value",  "Low Risk"):     ("D: UPSELL TARGET",
                                        "Stable — consider upsell or upgrade campaign"),
    }
    seg, action = mapping.get((value, risk), ("D: UPSELL TARGET", "Standard monitoring"))
    return risk, value, seg, action


# ── Main predict function ─────────────────────────────────────────
def predict_churn(customer_data: dict) -> dict:
    """
    Predict churn for a single customer.
    Args:
        customer_data: dict with all raw input fields
    Returns:
        dict with probability, prediction, tier, segment, action
    """
    pipeline  = get_pipeline()
    meta      = pipeline["metadata"]
    model     = pipeline["model"]
    encoders  = pipeline["encoders"]
    fe        = pipeline["fe_constants"]
    threshold = meta["decision_threshold"]
    features  = meta["features"]
    tiers     = meta["risk_tier_thresholds"]
    med       = fe["median_monthly_charges"]

    df_in   = pd.DataFrame([customer_data])
    df_feat = engineer_features(df_in, fe)
    df_enc  = encode_features(df_feat, encoders, features)

    prob = float(model.predict_proba(df_enc)[:, 1][0])
    pred = int(prob >= threshold)

    mc                    = float(customer_data.get("monthlycharges", 0))
    risk, value, seg, act = get_segment(prob, mc, tiers, med)

    return {
        "churn_probability":  round(prob, 4),
        "churn_predicted":    pred,
        "risk_tier":          risk,
        "value_tier":         value,
        "segment":            seg,
        "recommended_action": act,
        "threshold_used":     threshold,
    }
