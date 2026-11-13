
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data" / "churn_data.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "churn_scaler.pkl"

# Load data
df = pd.read_csv(INPUT_PATH)
df["SnapshotDate"] = pd.to_datetime(df["SnapshotDate"])

# Select features and target
features = ["Recency", "Frequency", "Monetary"]
X = df[features].copy()
y = df["Churn"]

# Log transform skewed features
X = np.log1p(X)

# Chronological split
unique_dates = sorted(df["SnapshotDate"].unique())
split_index = int(len(unique_dates) * 0.8)
split_date = unique_dates[split_index]

train_mask = df["SnapshotDate"] < split_date
test_mask = df["SnapshotDate"] >= split_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
# model = RandomForestClassifier(
#     n_estimators=200,
#     max_depth=12,
#     min_samples_split=10,
#     min_samples_leaf=4,
#     class_weight="balanced",
#     random_state=42,
#     n_jobs=-1
# )

# model.fit(X_train_scaled, y_train)

from sklearn.model_selection import RandomizedSearchCV

# Hyperparameter search space
param_grid = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [8, 10, 12, 15, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 6],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced", "balanced_subsample"]
}

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=20,
    scoring="roc_auc",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train_scaled, y_train)

model = search.best_estimator_

print("\nBest Parameters:")
print(search.best_params_)

print("\nBest Cross-Validation ROC-AUC:")
print(search.best_score_)


# Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("Split Date:", split_date)
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))

# Save model and scaler
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\nModel and scaler saved successfully!")

# Feature Importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

# Generate churn probabilities
churn_probability = model.predict_proba(X_test_scaled)[:, 1]

# Create evaluation dataframe
results = df.loc[test_mask].copy()

results["Churn_Probability"] = churn_probability

# Assign risk categories
def assign_risk(probability):
    if probability >= 0.70:
        return "High Risk"
    elif probability >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"

results["Risk_Category"] = results["Churn_Probability"].apply(
    assign_risk
)

print("\nChurn Risk Distribution:")
print(results["Risk_Category"].value_counts())

print("\nAverage Churn Probability by Risk:")
print(
    results.groupby("Risk_Category")["Churn_Probability"]
    .mean()
)

# Save predictions
OUTPUT_PATH = BASE_DIR / "data" / "churn_predictions.csv"
results.to_csv(OUTPUT_PATH, index=False)

print("\nPredictions saved to:", OUTPUT_PATH)