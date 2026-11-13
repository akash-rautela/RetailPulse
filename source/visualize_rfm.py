
import pandas as pd
import plotly.express as px
from pathlib import Path

# Load RFM dataset
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "rfm_data.csv"

rfm = pd.read_csv(INPUT_PATH)

# 1. Customer Segment Distribution
segment_counts = rfm["Segment"].value_counts().reset_index()
segment_counts.columns = ["Segment", "Customers"]

fig1 = px.bar(
    segment_counts,
    x="Segment",
    y="Customers",
    color="Segment",
    title="Customer Distribution by Segment",
    text="Customers"
)

fig1.update_layout(showlegend=False)
fig1.show()


# 2. Recency vs Frequency
fig2 = px.scatter(
    rfm,
    x="Recency",
    y="Frequency",
    color="Segment",
    hover_data=["CustomerID", "Monetary"],
    title="Customer Behaviour: Recency vs Frequency"
)

fig2.show()


# 3. Monetary Distribution
print("Monetary Data Information:")
print(rfm["Monetary"].describe())

print("Data Type:", rfm["Monetary"].dtype)
print("Missing Values:", rfm["Monetary"].isna().sum())
print("Total Customers:", len(rfm))


# 3. Customer Spending Distribution

rfm["Monetary"] = pd.to_numeric(
    rfm["Monetary"], errors="coerce"
)

spending_data = rfm.dropna(subset=["Monetary"])
spending_data = spending_data[spending_data["Monetary"] > 0]

fig3 = px.histogram(
    spending_data,
    x="Monetary",
    nbins=40,
    title="Customer Spending Distribution",
    labels={
        "Monetary": "Total Customer Spending",
        "count": "Number of Customers"
    }
)

fig3.update_layout(
    template="plotly_white",
    bargap=0.05,
    xaxis_title="Total Spending (£)",
    yaxis_title="Number of Customers"
)

fig3.show()

