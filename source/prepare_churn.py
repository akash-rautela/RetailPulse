
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data" / "cleaned_data.csv"
OUTPUT_PATH = BASE_DIR / "data" / "churn_data.csv"

df = pd.read_csv(INPUT_PATH)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Remove invalid customer IDs
df = df.dropna(subset=["CustomerID"])

# Ensure customer IDs are consistent
df["CustomerID"] = df["CustomerID"].astype(int)

# Define observation and prediction periods
lookback_days = 180
prediction_days = 90

# Dataset ends on 9 December 2011
last_date = df["InvoiceDate"].max()

# Latest snapshot must have a complete 90-day future window
latest_snapshot = last_date.normalize() - pd.Timedelta(
    days=prediction_days
)

# Create monthly snapshots
snapshots = pd.date_range(
    start="2011-03-31",
    end=latest_snapshot,
    freq="ME"
).tolist()

# Add the exact latest valid snapshot
if not snapshots or snapshots[-1] != latest_snapshot:
    snapshots.append(latest_snapshot)

churn_records = []

for snapshot in snapshots:

    # Historical data: previous 180 days
    history = df[
        (df["InvoiceDate"] <= snapshot) &
        (df["InvoiceDate"] > snapshot - pd.Timedelta(days=lookback_days))
    ]

    # Future data: next 90 days
    future = df[
        (df["InvoiceDate"] > snapshot) &
        (df["InvoiceDate"] <= snapshot + pd.Timedelta(days=prediction_days))
    ]

    # Calculate customer features
    features = history.groupby("CustomerID").agg(
        Recency=("InvoiceDate",
                 lambda x: (snapshot + pd.Timedelta(days=1) - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalAmount", "sum")
    )

    # Customers who purchase in the next 90 days
    future_customers = set(future["CustomerID"].unique())

    # Create churn labels
    features["Churn"] = (
        ~features.index.isin(future_customers)
    ).astype(int)

    features["SnapshotDate"] = snapshot

    churn_records.append(features.reset_index())

# Combine all snapshots
churn_df = pd.concat(churn_records, ignore_index=True)

# Save dataset
churn_df.to_csv(OUTPUT_PATH, index=False)

print("Churn Dataset Created Successfully!")

print("\nDataset Shape:", churn_df.shape)

print("\nChurn Distribution:")
print(churn_df["Churn"].value_counts())

print("\nChurn Percentage:")
print(churn_df["Churn"].value_counts(normalize=True) * 100)

print("\nDataset Preview:")
print(churn_df.head())

print("\nSaved to:", OUTPUT_PATH)
