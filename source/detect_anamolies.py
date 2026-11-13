
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import IsolationForest


# ==========================================================
# 1. PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data" / "cleaned_data.csv"
OUTPUT_PATH = BASE_DIR / "data" / "anomaly_results.csv"
MODEL_PATH = BASE_DIR / "models" / "anomaly_model.pkl"


# ==========================================================
# 2. LOAD DATA
# ==========================================================

print("Loading retail dataset...")

df = pd.read_csv(INPUT_PATH)

print("Original Dataset Shape:", df.shape)


# ==========================================================
# 3. FEATURE SELECTION
# ==========================================================

features = [
    "Quantity",
    "UnitPrice",
    "TotalAmount"
]

missing = [
    col for col in features
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")


# Convert features to numeric
for col in features:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Remove invalid values
df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=features
)

# Only positive purchase transactions
df = df[
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0) &
    (df["TotalAmount"] > 0)
].copy()

print("Valid Transactions:", len(df))


# ==========================================================
# 4. LOG TRANSFORMATION
# ==========================================================

X = df[features].copy()

# Reduces the impact of highly skewed transaction values
X = np.log1p(X)


# ==========================================================
# 5. TRAIN ISOLATION FOREST
# ==========================================================

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    max_samples="auto",
    random_state=42,
    n_jobs=-1
)

model.fit(X)


# ==========================================================
# 6. PREDICT ANOMALIES
# ==========================================================

print("Detecting unusual transactions...")

predictions = model.predict(X)

# Isolation Forest:
# -1 = Anomaly
#  1 = Normal

df["Anomaly"] = np.where(
    predictions == -1,
    "Anomaly",
    "Normal"
)

# Higher score = more unusual
df["Anomaly_Score"] = -model.decision_function(X)

df["Anomaly_Score"] = df["Anomaly_Score"].round(6)


# ==========================================================
# 7. SAVE RESULTS
# ==========================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

joblib.dump(
    model,
    MODEL_PATH
)


# ==========================================================
# 8. RESULTS
# ==========================================================

total = len(df)

anomalies = (
    df["Anomaly"] == "Anomaly"
).sum()

normal = (
    df["Anomaly"] == "Normal"
).sum()

print("\n" + "=" * 45)
print("ANOMALY DETECTION RESULTS")
print("=" * 45)

print("Total Transactions:", f"{total:,}")
print("Normal Transactions:", f"{normal:,}")
print("Anomalous Transactions:", f"{anomalies:,}")

print(
    "Anomaly Percentage:",
    f"{(anomalies / total) * 100:.2f}%"
)

print("\nTop 10 Unusual Transactions:")

print(
    df.sort_values(
        "Anomaly_Score",
        ascending=False
    )[
        [
            "InvoiceNo",
            "CustomerID",
            "Quantity",
            "UnitPrice",
            "TotalAmount",
            "Anomaly_Score"
        ]
    ].head(10).to_string(index=False)
)

print("\nResults saved:", OUTPUT_PATH)
print("Model saved:", MODEL_PATH)

print("\nAnomaly detection completed successfully!")
