import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

DATA_DIR = "data"
TRAIN_FILE = "google_ads_main_cols_fixed.csv"

df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILE), encoding="utf-8")

# Normalize column names
df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]

# Convert empty/placeholder values into NaN
df = df.replace(["--", "", None], np.nan)

# Convert CTR from percentage string into float
if "ctr" in df.columns:
    df["ctr"] = df["ctr"].astype(str).str.replace("%","",regex=False)
    df["ctr"] = pd.to_numeric(df["ctr"], errors="coerce") / 100.0

# Generate conversion label
if "conversions" in df.columns:
    df["converted"] = (pd.to_numeric(df["conversions"].fillna(0), errors="coerce") > 0).astype(int)
else:
    df["converted"] = 0

# Select training features
features = ["clicks","cost","avg_cpc","impressions","ctr"]
features = [f for f in features if f in df.columns]
X = df[features]
y = df["converted"]

# Preprocessing: fill missing numerical values
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="mean"), features)
])

# Build pipeline model
model = Pipeline([
    ("pre", pre),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        class_weight="balanced"
    ))
])

# Train model
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
model.fit(X_train, y_train)

# Save trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/conversion_model.pkl")

# Check if probability output is valid
if hasattr(model["clf"], "predict_proba"):
    y_prob = model.predict_proba(X_test)[:,1]
    print("\n Prediction probability examples:", np.round(y_prob[:10],3))
