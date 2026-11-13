
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data" / "rfm_data.csv"
OUTPUT_PATH = BASE_DIR / "data" / "clustered_customers.csv"

# Load RFM data
rfm = pd.read_csv(INPUT_PATH, index_col="CustomerID")

# Select RFM features
features = ["Recency", "Frequency", "Monetary"]
X = rfm[features].copy()

# Handle invalid values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# Log transformation
X_log = np.log1p(X)

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# Find the best K using silhouette score
scores = {}

for k in range(2, 8):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels,
        random_state=42
    )

    scores[k] = score
    print(f"K = {k}, Silhouette Score = {score:.4f}")

# Select K with the highest silhouette score
best_k = max(scores, key=scores.get)

print("\nBest K:", best_k)

# Train final K-Means model
kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(X_scaled)

# Save results
rfm.to_csv(OUTPUT_PATH)

# Analyze clusters
cluster_summary = rfm.groupby("Cluster")[features].mean().round(2)

print("\nCluster Summary:")
print(cluster_summary)

print("\nCustomers in Each Cluster:")
print(rfm["Cluster"].value_counts())

print("\nSaved to:", OUTPUT_PATH)
