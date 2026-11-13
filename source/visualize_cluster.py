
import pandas as pd
import plotly.express as px
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data" / "clustered_customers.csv"

df = pd.read_csv(INPUT_PATH, index_col="CustomerID")

df["Cluster"] = df["Cluster"].astype(str)

# 1. Customer distribution
fig1 = px.pie(
    df,
    names="Cluster",
    title="Customer Distribution by Cluster",
    hole=0.45
)

fig1.show()


# 2. Recency vs Monetary
fig2 = px.scatter(
    df,
    x="Recency",
    y="Monetary",
    color="Cluster",
    hover_data=["Frequency"],
    log_y=True,
    title="Customer Segmentation: Recency vs Spending"
)

fig2.show()


# 3. Frequency vs Monetary
fig3 = px.scatter(
    df,
    x="Frequency",
    y="Monetary",
    color="Cluster",
    hover_data=["Recency"],
    log_y=True,
    title="Customer Segmentation: Frequency vs Spending"
)

fig3.show()


# 4. Compare cluster averages
summary = df.groupby("Cluster")[
    ["Recency", "Frequency", "Monetary"]
].mean().reset_index()

summary_melted = summary.melt(
    id_vars="Cluster",
    var_name="Metric",
    value_name="Average Value"
)

fig4 = px.bar(
    summary_melted,
    x="Metric",
    y="Average Value",
    color="Cluster",
    barmode="group",
    title="Average RFM Metrics by Cluster"
)

fig4.show()
