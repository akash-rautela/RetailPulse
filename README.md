# RetailPulse — Customer Segmentation & Churn Intelligence

**An end-to-end Machine Learning project for retail analytics, customer segmentation, churn-risk prediction, and unusual transaction detection.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red?logo=streamlit)](https://retailpulse-akash.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)](https://scikit-learn.org/)

## Live Demo

**[Explore RetailPulse Dashboard](https://retailpulse-akash.streamlit.app/)**

## Overview

RetailPulse is an interactive retail intelligence dashboard built using Python and traditional Machine Learning algorithms.

It transforms transactional retail data into actionable business insights by identifying valuable customer segments, predicting customer inactivity risk, and detecting unusual transaction patterns.

The project uses the UCI Online Retail dataset and combines data preprocessing, exploratory analysis, feature engineering, model training, and interactive visualization.

## Key Features

### 1. Customer Segmentation
- RFM (Recency, Frequency, Monetary) analysis.
- K-Means clustering.
- Customer behaviour profiling.
- Cluster-level spending and purchasing analysis.

### 2. Churn Intelligence
- Predicts customer inactivity risk over a 90-day horizon.
- Random Forest classification.
- Temporal train-test evaluation.
- Customer-level risk categorization.
- Feature importance analysis.

### 3. Anomaly Detection
- Isolation Forest algorithm.
- Identifies statistically unusual transactions.
- Anomaly score distribution.
- Interactive transaction filtering.
- Customer ID search and CSV export.

### 4. Interactive Dashboard
- Business performance KPIs.
- Revenue and transaction insights.
- Interactive Plotly visualizations.
- Downloadable reports and filtered datasets.

## Machine Learning Models

| Task | Algorithm |
|---|---|
| Customer Segmentation | K-Means Clustering |
| Churn Prediction | Random Forest Classifier |
| Anomaly Detection | Isolation Forest |

## Model Performance

### Churn Prediction

| Metric | Result |
|---|---:|
| Accuracy | 66% |
| ROC-AUC | 0.731 |
| Churn Recall | 74% |
| Churn Precision | 54% |

The model prioritizes identifying potentially inactive customers, while accepting some false-positive predictions.

### Customer Segmentation

| Metric | Result |
|---|---:|
| Selected clusters | 2 |
| Silhouette Score | 0.4328 |
| Total customers | 4,338 |

## Technology Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **Model Persistence:** Joblib
- **Development:** VS Code
- **Deployment:** Streamlit Community Cloud

## Project Structure

```text
RetailPulse/
│
├── data/
│   ├── cleaned_data.csv
│   ├── rfm_data.csv
│   ├── clustered_customers.csv
│   ├── churn_data.csv
│   ├── churn_predictions.csv
│   └── anomaly_results.csv
│
├── models/
│   └── churn_model.pkl
│
├── source/
│   ├── clean_data.py
│   ├── inspect_data.py
│   ├── rfm_analysis.py
│   ├── cluster_customer.py
│   ├── prepare_churn.py
│   ├── train_churn_model.py
│   ├── detect_anamolies.py
│   └── visualize_cluster.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/akash-rautela/RetailPulse.git
cd RetailPulse
```

2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate it:

**Windows:**
```bash
.venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the dashboard:

```bash
streamlit run app.py
```

## Dataset

**UCI Online Retail Dataset**

The project uses the Online Retail transactional dataset containing retail purchases from December 2010 to December 2011.

Source: [UCI Machine Learning Repository — Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)

## Important Notes

- Churn represents predicted 90-day inactivity risk, not confirmed permanent customer loss.
- Anomaly detection identifies statistically unusual transactions; these are not necessarily fraudulent.
- Model evaluation results are based on the project's temporal test split.
- The deployed dashboard uses preprocessed data and saved model outputs.

## Future Improvements

- Customer-level retention recommendations.
- Automated customer segmentation updates.
- Advanced churn prediction models.
- Real-time transaction anomaly scoring.
- Automated PDF business reports.

## Author

**Akash Rautela**

- GitHub: [@akash-rautela](https://github.com/akash-rautela)
- Project: [RetailPulse](https://github.com/akash-rautela/RetailPulse)
- Live Demo: [retailpulse-akash.streamlit.app](https://retailpulse-akash.streamlit.app/)

---

*Built with Python, Machine Learning, and Streamlit.*