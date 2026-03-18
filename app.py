# app.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

DATA_DIR = "data"
MODEL_PATH = os.path.join("models", "conversion_model.pkl")

google_path = os.path.join(DATA_DIR, "google_ads_main_cols_fixed.csv")
schedule_path = os.path.join(DATA_DIR, "schedule_v1_cols_fixed.csv")
age_path = os.path.join(DATA_DIR, "age_report_cols_fixed.csv")
fb_path = os.path.join(DATA_DIR, "facebook_ads_cols_fixed.csv")

df_google = pd.read_csv(google_path)
df_schedule = pd.read_csv(schedule_path)
df_age = pd.read_csv(age_path)
df_fb = pd.read_csv(fb_path)

# google_ads_main: Day, Clicks, Impressions, CTR, Currency Code, Avg CPC, Cost, Conversions, Conv rate, ...
total_clicks = pd.to_numeric(df_google["Clicks"], errors="coerce").fillna(0).sum()
total_impr = pd.to_numeric(df_google["Impressions"], errors="coerce").fillna(0).sum()
total_cost = pd.to_numeric(df_google["Cost"], errors="coerce").fillna(0).sum()

# conversion column
conv_col_google = "Conversions" if "Conversions" in df_google.columns else "Conversions"
total_conv = pd.to_numeric(df_google.get(conv_col_google, 0), errors="coerce").fillna(0).sum()

avg_cpc_kpi = (total_cost / total_clicks) if total_clicks > 0 else 0
ctr_kpi = (total_clicks / total_impr * 100) if total_impr > 0 else 0
conv_rate_kpi = (total_conv / total_clicks * 100) if total_clicks > 0 else 0

df_google_trend = df_google.copy()
# Day is already a YYYY-MM-DD string, so we can sort directly
df_google_trend = df_google_trend.sort_values("Day")

line_dates = df_google_trend["Day"].astype(str).tolist()
line_clicks = pd.to_numeric(df_google_trend["Clicks"], errors="coerce").fillna(0).tolist()
line_cost = pd.to_numeric(df_google_trend["Cost"], errors="coerce").fillna(0).tolist()

# schedule: Day of the week, Hour of the day, Clicks, Impressions, CTR, Currency Code, Avg CPC, Cost, Conversion Rate, Conversions, Cost _ conv
df_heat = df_schedule.copy()
df_heat["Hour of the day"] = pd.to_numeric(df_heat["Hour of the day"], errors="coerce")
df_heat["Clicks"] = pd.to_numeric(df_heat["Clicks"], errors="coerce").fillna(0)

# fixed weekday order
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# keep only the weekdays that exist in the dataset
days_in_data = [d for d in weekday_order if d in df_heat["Day of the week"].unique().tolist()]
if not days_in_data:
    days_in_data = sorted(df_heat["Day of the week"].astype(str).unique().tolist())

hours_in_data = sorted(df_heat["Hour of the day"].dropna().unique().tolist())

# aggregate clicks
pivot = (
    df_heat.groupby(["Day of the week", "Hour of the day"])["Clicks"]
    .sum()
    .reset_index()
)

heatmap_data = []
for _, row in pivot.iterrows():
    day = row["Day of the week"]
    hour = row["Hour of the day"]
    val = row["Clicks"]
    if day in days_in_data and hour in hours_in_data:
        x_idx = hours_in_data.index(hour)
        y_idx = days_in_data.index(day)
        heatmap_data.append([x_idx, y_idx, float(val)])

# age_report: Demographic status, Age, Ad group, Status, Conversions, Currency Code, Cost _ conv, Bid adj, Clicks, Impressions, CTR, Avg CPC, Cost
df_age_plot = df_age.copy()
df_age_plot["Clicks"] = pd.to_numeric(df_age_plot["Clicks"], errors="coerce").fillna(0)
# if Conversions column exists, use conversion count; otherwise, use Clicks as fallback
if "Conversions" in df_age_plot.columns:
    df_age_plot["Conversions"] = pd.to_numeric(df_age_plot["Conversions"], errors="coerce").fillna(0)
    age_metric = df_age_plot.groupby("Age")["Conversions"].sum().reset_index()
else:
    age_metric = df_age_plot.groupby("Age")["Clicks"].sum().reset_index()

age_labels = age_metric["Age"].astype(str).tolist()
age_values = age_metric.iloc[:, 1].tolist()

# ========== 6. Channel Spend Pie Chart (Google vs Facebook) ==========
google_spend = pd.to_numeric(df_google["Cost"], errors="coerce").fillna(0).sum()

# facebook_ads_main: Amount spent CAD
fb_spend = pd.to_numeric(df_fb["Amount spent CAD"], errors="coerce").fillna(0).sum()

channel_labels = ["Google Ads", "Facebook Ads"]
channel_spend = [float(google_spend), float(fb_spend)]

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    print(f"⚠ Model file not found: {MODEL_PATH}, prediction API will be disabled")

@app.route("/")
def index():
    return render_template(
        "index.html",
        # KPI metrics
        avg_cpc_kpi=round(avg_cpc_kpi, 2),
        ctr_kpi=round(ctr_kpi, 2),
        conv_rate_kpi=round(conv_rate_kpi, 2),
        total_clicks=int(total_clicks),
        total_impr=int(total_impr),
        total_cost=round(total_cost, 2),
        # Line chart data
        line_dates=json.dumps(line_dates),
        line_clicks=json.dumps(line_clicks),
        line_cost=json.dumps(line_cost),
        # Heatmap data
        heatmap_days=json.dumps(days_in_data),
        heatmap_hours=json.dumps(hours_in_data),
        heatmap_data=json.dumps(heatmap_data),
        # Age distribution chart
        age_labels=json.dumps(age_labels),
        age_values=json.dumps(age_values),
        # Channel spend pie chart
        channel_labels=json.dumps(channel_labels),
        channel_spend=json.dumps(channel_spend),
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    global model
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json(force=True)

    # Receive input from frontend
    clicks = float(data.get("clicks", 0) or 0)
    avg_cpc = float(data.get("avg_cpc", 0) or 0)
    impressions = float(data.get("impressions", 0) or 0)
    ctr = float(data.get("ctr", 0) or 0)

    # Support input like 12.5 (%) or 0.125 (ratio)
    if ctr > 1:
        ctr = ctr / 100.0

    # Cost can be empty; if so, estimate using clicks * avg_cpc
    cost = data.get("cost", None)
    if cost is None or str(cost).strip() == "":
        cost = clicks * avg_cpc
    else:
        cost = float(cost)

    X = pd.DataFrame([{
        "clicks": clicks,
        "cost": cost,
        "avg_cpc": avg_cpc,
        "impressions": impressions,
        "ctr": ctr
    }])

    prob = float(model.predict_proba(X)[0, 1])
    label = int(prob >= 0.5)

    return jsonify({
        "success": True,
        # Prediction results
        "probability": round(prob, 4),
        "label": label
    })

if __name__ == "__main__":
    # Run Flask app
    app.run(debug=True)
