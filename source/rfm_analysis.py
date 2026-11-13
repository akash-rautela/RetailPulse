
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "cleaned_data.csv"
OUTPUT_PATH = BASE_DIR / "data" / "rfm_data.csv"

# Load cleaned data
df = pd.read_csv(INPUT_PATH)

# Convert InvoiceDate into datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Reference date: one day after the last transaction
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# Calculate RFM metrics
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (reference_date - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalAmount": "sum"
})

# Rename columns
rfm.rename(columns={
    "InvoiceDate": "Recency",
    "InvoiceNo": "Frequency",
    "TotalAmount": "Monetary"
}, inplace=True)

# Save RFM dataset
rfm.to_csv(OUTPUT_PATH)

print("RFM Analysis Completed!")
print("\nRFM Dataset Preview:")
print(rfm.head())

print("\nRFM Statistics:")
print(rfm.describe())

print("\nTotal Customers:", len(rfm))
print("\nSaved to:", OUTPUT_PATH)


# Calculate RFM scores (1 to 5)

rfm["R_Score"] = pd.qcut(
    rfm["Recency"],
    q=5,
    labels=[5, 4, 3, 2, 1]
).astype(int)

rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    q=5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["M_Score"] = pd.qcut(
    rfm["Monetary"],
    q=5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

# Combine scores
rfm["RFM_Score"] = (
    rfm["R_Score"].astype(str) +
    rfm["F_Score"].astype(str) +
    rfm["M_Score"].astype(str)
)

print("\nRFM Scoring Preview:")
print(rfm.head(10))


def segment_customer(row):

    if row["R_Score"] >= 4 and row["F_Score"] >= 4:
        return "Champions"

    elif row["R_Score"] >= 3 and row["F_Score"] >= 3:
        return "Loyal Customers"

    elif row["R_Score"] >= 4 and row["F_Score"] <= 2:
        return "New Customers"

    elif row["R_Score"] <= 2 and row["F_Score"] >= 3:
        return "At Risk"

    elif row["R_Score"] <= 2 and row["F_Score"] <= 2:
        return "Hibernating"

    else:
        return "Potential Loyalists"


rfm["Segment"] = rfm.apply(segment_customer, axis=1)

# Save final RFM dataset
rfm.to_csv(OUTPUT_PATH)

print("\nCustomer Segment Distribution:")
print(rfm["Segment"].value_counts())

print("\nFinal RFM Dataset:")
print(rfm.head(10))
