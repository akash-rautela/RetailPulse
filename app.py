
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ==========================================================
# 1. PATHS AND PAGE CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="RetailPulse | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 2. DARK SAAS THEME
# ==========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0B1020;
    color: #E5E7EB;
}

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #263247;
}

[data-testid="stMetric"] {
    background-color: #151D32;
    border: 1px solid #283047;
    padding: 18px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    color: #9CA3AF;
}

[data-testid="stMetricValue"] {
    color: #F9FAFB;
}

h1, h2, h3 {
    color: #F9FAFB;
}

p, label {
    color: #D1D5DB;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

hr {
    border-color: #283047;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. DATA LOADING
# ==========================================================

@st.cache_data
def load_data():

    files = {
        "retail": "cleaned_data.csv",
        "rfm": "rfm_data.csv",
        "clusters": "clustered_customers.csv",
        "churn_data": "churn_data.csv",
        "churn": "churn_predictions.csv",
        "anomalies": "anomaly_results.csv"
    }

    data = {}

    for key, filename in files.items():

        file_path = DATA_DIR / filename

        if file_path.exists():

            data[key] = pd.read_csv(file_path)

        else:

            data[key] = None

    return data


data = load_data()

retail = data["retail"]
rfm = data["rfm"]
clusters = data["clusters"]
churn_data = data["churn_data"]
churn = data["churn"]
anomalies = data["anomalies"]

# ==========================================================
# 4. DATA PREPARATION
# ==========================================================

if retail is not None:

    retail = retail.copy()

    if "InvoiceDate" in retail.columns:
        retail["InvoiceDate"] = pd.to_datetime(
            retail["InvoiceDate"],
            errors="coerce"
        )

    if "TotalAmount" in retail.columns:
        retail["TotalAmount"] = pd.to_numeric(
            retail["TotalAmount"],
            errors="coerce"
        )


if rfm is not None:
    rfm = rfm.copy()

if clusters is not None:
    clusters = clusters.copy()

if churn is not None:
    churn = churn.copy()


# ==========================================================
# 5. REUSABLE FUNCTIONS
# ==========================================================

def format_currency(value):
    return f"£{value:,.2f}"


def show_missing_data(filename):

    st.error(
        f"Required file '{filename}' was not found "
        "inside the data folder."
    )


def get_cluster_column(df):

    possible_names = [
        "Cluster",
        "cluster",
        "KMeans_Cluster",
        "KMeans Cluster"
    ]

    for name in possible_names:

        if name in df.columns:
            return name

    return None


def get_customer_column(df):

    possible_names = [
        "CustomerID",
        "Customer Id",
        "customer_id",
        "Customer ID"
    ]

    for name in possible_names:

        if name in df.columns:
            return name

    return None


# ==========================================================
# 6. SIDEBAR NAVIGATION
# ==========================================================

with st.sidebar:

    st.markdown("# 📊 RetailPulse")

    st.caption("Customer Intelligence Platform")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Customer Segmentation",
            "Churn Intelligence",
            "Anomaly Detection",
            "Reports & Insights"
        ]
    )

    st.divider()

    st.caption("ML-Powered Retail Analytics")
    st.caption("Version 1.0")


# ==========================================================
# 7. OVERVIEW PAGE
# ==========================================================

if page == "Overview":

    st.title("Overview")

    st.caption(
        "Monitor customer behaviour, revenue "
        "and retention performance."
    )

    st.divider()

    if retail is None:
        show_missing_data("cleaned_data.csv")
        st.stop()

    required_columns = [
        "CustomerID",
        "InvoiceNo",
        "InvoiceDate",
        "TotalAmount"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in retail.columns
    ]

    if missing_columns:

        st.error(
            f"Missing columns in cleaned_data.csv: "
            f"{missing_columns}"
        )

        st.stop()

    valid_retail = retail.dropna(
        subset=[
            "CustomerID",
            "InvoiceNo",
            "InvoiceDate",
            "TotalAmount"
        ]
    ).copy()

    # ------------------------------------------
    # BUSINESS METRICS
    # ------------------------------------------

    total_customers = valid_retail["CustomerID"].nunique()

    total_orders = valid_retail["InvoiceNo"].nunique()

    total_revenue = valid_retail["TotalAmount"].sum()

    average_order_value = (
        total_revenue / total_orders
        if total_orders > 0 else 0
    )

    # ------------------------------------------
    # KPI CARDS
    # ------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    col3.metric(
        "Total Revenue",
        format_currency(total_revenue)
    )

    col4.metric(
        "Average Order Value",
        format_currency(average_order_value)
    )

    st.divider()

    # ------------------------------------------
    # MONTHLY REVENUE
    # ------------------------------------------

    st.subheader("Revenue Trend")

    valid_retail["Month"] = (
        valid_retail["InvoiceDate"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly_revenue = (
        valid_retail.groupby("Month")["TotalAmount"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly_revenue,
        x="Month",
        y="TotalAmount",
        markers=True,
        template="plotly_dark",
        labels={
            "Month": "Month",
            "TotalAmount": "Revenue (£)"
        }
    )

    fig.update_traces(
        line=dict(color="#8174F7", width=3),
        marker=dict(size=7)
    )

    fig.update_layout(
        paper_bgcolor="#0B1020",
        plot_bgcolor="#151D32",
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title=None
    )
    st.caption(
    "Note: December 2011 contains partial-month data "
    "(transactions available through 9 December)."
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ------------------------------------------
    # CHURN RISK SNAPSHOT
    # ------------------------------------------

    st.subheader("Customer Retention Snapshot")

    if churn is not None and "Risk_Category" in churn.columns:

        risk_counts = (
            churn["Risk_Category"]
            .value_counts()
            .reindex(
                ["High Risk", "Medium Risk", "Low Risk"],
                fill_value=0
            )
            .reset_index()
        )

        risk_counts.columns = [
            "Risk Category",
            "Records"
        ]

        col1, col2 = st.columns(2)

        with col1:

            fig = px.pie(
                risk_counts,
                names="Risk Category",
                values="Records",
                hole=0.65,
                color="Risk Category",
                color_discrete_map={
                    "High Risk": "#F87171",
                    "Medium Risk": "#FBBF24",
                    "Low Risk": "#34D399"
                },
                template="plotly_dark"
            )

            fig.update_layout(
                paper_bgcolor="#0B1020",
                plot_bgcolor="#0B1020"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            st.markdown("### Risk Distribution")

            for _, row in risk_counts.iterrows():

                st.write(
                    f"**{row['Risk Category']}** — "
                    f"{row['Records']:,} records"
                )

            st.caption(
                "Historical test-snapshot predictions. "
                "A customer may appear in multiple snapshots."
            )

    else:

        st.info("Churn predictions are not available.")



# ==========================================================
# 8. CUSTOMER SEGMENTATION PAGE
# ==========================================================

elif page == "Customer Segmentation":

    st.title("Customer Segmentation")

    st.caption(
        "Understand customer behaviour through "
        "K-Means clustering and RFM analysis."
    )

    st.divider()

    # ------------------------------------------------------
    # LOAD AND VALIDATE DATA
    # ------------------------------------------------------

    if clusters is None:
        show_missing_data("clustered_customers.csv")
        st.stop()

    cluster_column = get_cluster_column(clusters)
    customer_column = get_customer_column(clusters)

    if cluster_column is None:

        st.error("Cluster column not found.")

        st.write("Available columns:", clusters.columns.tolist())
        st.stop()

    df = clusters.copy()

    # Convert numerical features
    for col in ["Recency", "Frequency", "Monetary"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    rfm_columns = [
        col for col in ["Recency", "Frequency", "Monetary"]
        if col in df.columns
    ]

    if not rfm_columns:
        st.error("RFM columns are missing from the dataset.")
        st.stop()

    df = df.dropna(subset=[cluster_column])

    # ------------------------------------------------------
    # TOP KPI CARDS
    # ------------------------------------------------------

    total_customers = (
        df[customer_column].nunique()
        if customer_column else len(df)
    )

    total_clusters = df[cluster_column].nunique()

    average_frequency = (
        df["Frequency"].mean()
        if "Frequency" in df.columns else 0
    )

    average_monetary = (
        df["Monetary"].mean()
        if "Monetary" in df.columns else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Customer Clusters",
        f"{total_clusters}"
    )

    col3.metric(
        "Avg. Purchase Frequency",
        f"{average_frequency:.2f}"
    )

    col4.metric(
        "Avg. Customer Spend",
        f"£{average_monetary:,.2f}"
    )

    st.divider()

    # ------------------------------------------------------
    # CLUSTER DISTRIBUTION
    # ------------------------------------------------------

    st.subheader("Customer Distribution")

    cluster_counts = (
        df[cluster_column]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_counts.columns = [
        "Cluster",
        "Customers"
    ]

    col1, col2 = st.columns([1, 1])

    with col1:

        fig = px.pie(
            cluster_counts,
            names="Cluster",
            values="Customers",
            hole=0.65,
            template="plotly_dark",
            title="Cluster Share"
        )

        fig.update_layout(
            paper_bgcolor="#0B1020",
            plot_bgcolor="#0B1020"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            cluster_counts,
            x="Cluster",
            y="Customers",
            color="Cluster",
            text="Customers",
            template="plotly_dark",
            title="Customers per Cluster"
        )

        fig.update_layout(
            paper_bgcolor="#0B1020",
            plot_bgcolor="#0B1020",
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # ------------------------------------------------------
    # CLUSTER BEHAVIOUR
    # ------------------------------------------------------

    st.subheader("Cluster Behaviour Analysis")

    cluster_summary = (
        df.groupby(cluster_column)[rfm_columns]
        .mean()
        .round(2)
    )

    st.markdown("#### Average RFM Metrics")

    st.dataframe(
        cluster_summary,
        use_container_width=True
    )

    # ------------------------------------------------------
    # AUTOMATIC CLUSTER PROFILES
    # ------------------------------------------------------

    st.subheader("Customer Cluster Profiles")

    profile_data = cluster_summary.copy()

    # Rank each cluster based on RFM behaviour.
    # Lower recency is better; higher frequency
    # and monetary values are better.

    scores = pd.DataFrame(index=profile_data.index)

    if "Recency" in profile_data.columns:
        scores["Recency"] = profile_data["Recency"].rank(
            ascending=False,
            pct=True
        )

    if "Frequency" in profile_data.columns:
        scores["Frequency"] = profile_data["Frequency"].rank(
            ascending=True,
            pct=True
        )

    if "Monetary" in profile_data.columns:
        scores["Monetary"] = profile_data["Monetary"].rank(
            ascending=True,
            pct=True
        )

    profile_data["Engagement Score"] = scores.mean(axis=1)

    best_cluster = profile_data[
        "Engagement Score"
    ].idxmax()

    profile_cards = st.columns(
        min(len(profile_data), 3)
    )

    for i, (cluster_id, row) in enumerate(
        profile_data.iterrows()
    ):

        with profile_cards[i % len(profile_cards)]:

            is_best = cluster_id == best_cluster

            profile_name = (
                "High-value / Engaged"
                if is_best
                else "Low-engagement"
            )

            st.markdown(
                f"### Cluster {cluster_id}"
            )

            st.caption(profile_name)

            st.metric(
                "Customers",
                f"{int(cluster_counts.loc[
                    cluster_counts['Cluster'] == cluster_id,
                    'Customers'
                ].iloc[0]):,}"
            )

            if "Recency" in row:
                st.write(
                    f"**Avg. Recency:** "
                    f"{row['Recency']:.1f} days"
                )

            if "Frequency" in row:
                st.write(
                    f"**Avg. Frequency:** "
                    f"{row['Frequency']:.2f}"
                )

            if "Monetary" in row:
                st.write(
                    f"**Avg. Monetary:** "
                    f"£{row['Monetary']:,.2f}"
                )

            st.caption(
                "Profile is relative to the other clusters."
            )

    st.divider()

    # ------------------------------------------------------
    # RFM SCATTER PLOT
    # ------------------------------------------------------

    st.subheader("Customer Behaviour Visualization")

    if "Recency" in df.columns and "Monetary" in df.columns:

        fig = px.scatter(
            df,
            x="Recency",
            y="Monetary",
            color=cluster_column,
            hover_data=[
                col for col in [
                    customer_column,
                    "Frequency"
                ]
                if col and col in df.columns
            ],
            template="plotly_dark",
            title="Recency vs Monetary by Cluster",
            opacity=0.65
        )

        fig.update_layout(
            paper_bgcolor="#0B1020",
            plot_bgcolor="#151D32",
            xaxis_title="Recency (Days)",
            yaxis_title="Monetary Value (£)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if "Frequency" in df.columns and "Monetary" in df.columns:

        fig = px.scatter(
            df,
            x="Frequency",
            y="Monetary",
            color=cluster_column,
            hover_data=[
                customer_column
            ] if customer_column else None,
            template="plotly_dark",
            title="Frequency vs Monetary",
            opacity=0.65
        )

        fig.update_layout(
            paper_bgcolor="#0B1020",
            plot_bgcolor="#151D32"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # ------------------------------------------------------
    # CUSTOMER FILTERING
    # ------------------------------------------------------

    st.subheader("Explore Customer Records")

    available_clusters = sorted(
        df[cluster_column].unique().tolist(),
        key=str
    )

    selected_cluster = st.selectbox(
        "Select Customer Cluster",
        ["All Clusters"] + available_clusters
    )

    filtered_df = df.copy()

    if selected_cluster != "All Clusters":

        filtered_df = filtered_df[
            filtered_df[cluster_column] == selected_cluster
        ]

    if customer_column:

        search_customer = st.text_input(
            "Search Customer ID",
            placeholder="Enter Customer ID..."
        )

        if search_customer.strip():

            filtered_df = filtered_df[
                filtered_df[customer_column]
                .astype(str)
                .str.contains(
                    search_customer.strip(),
                    case=False,
                    na=False
                )
            ]

    st.write(
        f"Showing {len(filtered_df):,} customer records"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    # ------------------------------------------------------
    # DOWNLOAD CLUSTER DATA
    # ------------------------------------------------------

    st.download_button(
        label="Download Clustered Customer Data",
        data=filtered_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="customer_segments.csv",
        mime="text/csv"
    )



# ==========================================================
# 9. CHURN INTELLIGENCE PAGE
# ==========================================================

elif page == "Churn Intelligence":

    import joblib

    st.title("Churn Intelligence")

    st.caption(
        "Identify customers at risk of becoming inactive "
        "within the next 90 days."
    )

    st.divider()

    # ------------------------------------------------------
    # LOAD AND VALIDATE DATA
    # ------------------------------------------------------

    if churn is None:
        show_missing_data("churn_predictions.csv")
        st.stop()

    required_columns = [
        "CustomerID",
        "SnapshotDate",
        "Churn_Probability",
        "Risk_Category"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in churn.columns
    ]

    if missing_columns:
        st.error(f"Missing columns: {missing_columns}")
        st.write("Available columns:", churn.columns.tolist())
        st.stop()

    df = churn.copy()

    df["SnapshotDate"] = pd.to_datetime(
        df["SnapshotDate"],
        errors="coerce"
    )

    df["Churn_Probability"] = pd.to_numeric(
        df["Churn_Probability"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "CustomerID",
            "SnapshotDate",
            "Churn_Probability",
            "Risk_Category"
        ]
    )

    if df.empty:
        st.warning("No valid churn prediction records found.")
        st.stop()

    risk_order = [
        "High Risk",
        "Medium Risk",
        "Low Risk"
    ]

    # ------------------------------------------------------
    # FILTERS
    # ------------------------------------------------------

    st.subheader("Prediction Filters")

    filter_col1, filter_col2 = st.columns(2)

    available_dates = sorted(
        df["SnapshotDate"].dt.strftime("%Y-%m-%d").unique()
    )

    with filter_col1:

        selected_date = st.selectbox(
            "Snapshot Date",
            ["All Dates"] + available_dates,
            index=len(available_dates)
        )

    with filter_col2:

        selected_risk = st.selectbox(
            "Risk Category",
            ["All Risks"] + risk_order
        )

    filtered_df = df.copy()

    if selected_date != "All Dates":

        filtered_df = filtered_df[
            filtered_df["SnapshotDate"].dt.strftime("%Y-%m-%d")
            == selected_date
        ]

    if selected_risk != "All Risks":

        filtered_df = filtered_df[
            filtered_df["Risk_Category"] == selected_risk
        ]

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        st.stop()

    # ------------------------------------------------------
    # KPI CARDS
    # ------------------------------------------------------

    st.subheader("Customer Risk Overview")

    total_records = len(filtered_df)

    high_risk = (
        filtered_df["Risk_Category"] == "High Risk"
    ).sum()

    medium_risk = (
        filtered_df["Risk_Category"] == "Medium Risk"
    ).sum()

    low_risk = (
        filtered_df["Risk_Category"] == "Low Risk"
    ).sum()

    average_probability = (
        filtered_df["Churn_Probability"].mean() * 100
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Prediction Records",
        f"{total_records:,}"
    )

    col2.metric(
        "High Risk",
        f"{high_risk:,}"
    )

    col3.metric(
        "Medium Risk",
        f"{medium_risk:,}"
    )

    col4.metric(
        "Average Risk Probability",
        f"{average_probability:.1f}%"
    )

    st.caption(
        "Counts represent prediction records. "
        "The same customer can appear in multiple snapshots."
    )

    st.divider()

    # ------------------------------------------------------
    # RISK DISTRIBUTION
    # ------------------------------------------------------

    st.subheader("Risk Distribution")

    risk_counts = (
        filtered_df["Risk_Category"]
        .value_counts()
        .reindex(risk_order, fill_value=0)
        .reset_index()
    )

    risk_counts.columns = [
        "Risk Category",
        "Records"
    ]

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            risk_counts,
            names="Risk Category",
            values="Records",
            hole=0.65,
            color="Risk Category",
            color_discrete_map={
                "High Risk": "#F87171",
                "Medium Risk": "#FBBF24",
                "Low Risk": "#34D399"
            },
            template="plotly_dark",
            title="Customer Risk Composition"
        )

        fig.update_layout(
            paper_bgcolor="#0B1020",
            plot_bgcolor="#0B1020"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.histogram(
            filtered_df,
            x="Churn_Probability",
            nbins=25,
            color="Risk_Category",
            color_discrete_map={
                "High Risk": "#F87171",
                "Medium Risk": "#FBBF24",
                "Low Risk": "#34D399"
            },
            template="plotly_dark",
            title="Churn Probability Distribution",
            labels={
                "Churn_Probability": "Predicted Inactivity Probability"
            }
        )

        fig.update_layout(
            paper_bgcolor="#0B1020",
            plot_bgcolor="#151D32",
            bargap=0.08
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # ------------------------------------------------------
    # RISK CATEGORY PERFORMANCE
    # ------------------------------------------------------

    st.subheader("Risk Category Performance")

    if "Churn" in filtered_df.columns:

        performance = (
            filtered_df.groupby("Risk_Category")
            .agg(
                Prediction_Records=("Churn", "size"),
                Average_Probability=("Churn_Probability", "mean"),
                Actual_Inactivity_Rate=("Churn", "mean")
            )
            .reindex(risk_order)
            .dropna(how="all")
            .reset_index()
        )

        performance["Average_Probability"] = (
            performance["Average_Probability"] * 100
        ).round(2)

        performance["Actual_Inactivity_Rate"] = (
            performance["Actual_Inactivity_Rate"] * 100
        ).round(2)

        performance.rename(
            columns={
                "Prediction_Records": "Records",
                "Average_Probability": "Avg Predicted Risk (%)",
                "Actual_Inactivity_Rate": "Observed Inactivity (%)"
            },
            inplace=True
        )

        st.dataframe(
            performance,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Observed inactivity is calculated from the historical "
            "90-day target labels. It is not a future live outcome."
        )

    else:

        st.info("Actual churn labels are unavailable.")

    st.divider()

    # ------------------------------------------------------
    # INDIVIDUAL CUSTOMER LOOKUP
    # ------------------------------------------------------

    st.subheader("Individual Customer Analysis")

    customer_search = st.text_input(
        "Search Customer ID",
        placeholder="Example: 12346"
    )

    if customer_search.strip():

        customer_records = df[
            df["CustomerID"].astype(str).str.strip()
            == customer_search.strip()
        ].copy()

        if customer_records.empty:

            st.warning("Customer not found in prediction records.")

        else:

            # Use the latest available prediction for this customer
            customer_records = customer_records.sort_values(
                "SnapshotDate"
            )

            customer = customer_records.iloc[-1]

            st.markdown("### Customer Profile")

            profile_col1, profile_col2, profile_col3 = st.columns(3)

            profile_col1.metric(
                "Customer ID",
                str(customer["CustomerID"])
            )

            profile_col2.metric(
                "Predicted Inactivity Risk",
                f"{customer['Churn_Probability'] * 100:.2f}%"
            )

            profile_col3.metric(
                "Risk Category",
                customer["Risk_Category"]
            )

            # Risk indicator
            probability = float(customer["Churn_Probability"])

            st.progress(
                min(max(probability, 0.0), 1.0),
                text=f"Predicted risk: {probability * 100:.1f}%"
            )

            # Customer RFM details
            metric_col1, metric_col2, metric_col3 = st.columns(3)

            if "Recency" in customer.index:

                metric_col1.metric(
                    "Recency",
                    f"{customer['Recency']} days"
                )

            if "Frequency" in customer.index:

                metric_col2.metric(
                    "Frequency",
                    f"{customer['Frequency']}"
                )

            if "Monetary" in customer.index:

                metric_col3.metric(
                    "Monetary",
                    f"£{customer['Monetary']:,.2f}"
                )

            st.write(
                f"Latest prediction snapshot: "
                f"{customer['SnapshotDate'].date()}"
            )

            # Customer history
            st.markdown("#### Prediction History")

            history_fig = px.line(
                customer_records,
                x="SnapshotDate",
                y="Churn_Probability",
                markers=True,
                template="plotly_dark",
                title="Historical Inactivity Risk"
            )

            history_fig.update_layout(
                paper_bgcolor="#0B1020",
                plot_bgcolor="#151D32",
                yaxis_tickformat=".0%",
                yaxis_title="Predicted Risk"
            )

            st.plotly_chart(
                history_fig,
                use_container_width=True
            )

    st.divider()

    # ------------------------------------------------------
    # RANDOM FOREST FEATURE IMPORTANCE
    # ------------------------------------------------------

    st.subheader("Model Insights")

    model_path = MODEL_DIR / "churn_model.pkl"

    if model_path.exists():

        try:

            model = joblib.load(model_path)

            if hasattr(model, "feature_importances_"):

                feature_names = [
                    "Recency",
                    "Frequency",
                    "Monetary"
                ]

                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": model.feature_importances_
                }).sort_values(
                    "Importance",
                    ascending=True
                )

                fig = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    text="Importance",
                    template="plotly_dark",
                    title="Random Forest Feature Importance"
                )

                fig.update_traces(
                    marker_color="#8174F7",
                    texttemplate="%{x:.2%}"
                )

                fig.update_layout(
                    paper_bgcolor="#0B1020",
                    plot_bgcolor="#151D32",
                    xaxis_tickformat=".0%"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.info(
                    "Feature importance is unavailable for this model."
                )

        except Exception as error:

            st.warning(
                f"Could not load the saved model: {error}"
            )

    else:

        st.info(
            "Saved churn_model.pkl was not found in the models folder."
        )

    st.divider()

    # ------------------------------------------------------
    # CUSTOMER RECORD TABLE
    # ------------------------------------------------------

    st.subheader("Customer Risk Records")

    display_columns = [
        col for col in [
            "CustomerID",
            "SnapshotDate",
            "Recency",
            "Frequency",
            "Monetary",
            "Churn_Probability",
            "Risk_Category"
        ]
        if col in filtered_df.columns
    ]

    sorted_records = filtered_df.sort_values(
        "Churn_Probability",
        ascending=False
    )

    st.dataframe(
        sorted_records[display_columns].head(100),
        use_container_width=True,
        hide_index=True
    )

    # ------------------------------------------------------
    # DOWNLOAD
    # ------------------------------------------------------

    st.download_button(
        label="Download Filtered Predictions",
        data=filtered_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="filtered_churn_predictions.csv",
        mime="text/csv"
    )

    st.caption(
        "Risk categories use the project's initial probability thresholds: "
        "High ≥ 70%, Medium ≥ 40%, Low < 40%. "
        "Probabilities are model estimates, not guarantees."
    )


# ==========================================================
# ANOMALY DETECTION PAGE
# ==========================================================


elif page == "Anomaly Detection":

    st.title("Anomaly Detection")

    st.caption(
        "Identify unusual transaction patterns "
        "using the Isolation Forest algorithm."
    )

    st.divider()

    if anomalies is None:
        show_missing_data("anomaly_results.csv")
        st.stop()

    df = anomalies.copy()

    required_columns = [
        "Anomaly",
        "Anomaly_Score",
        "Quantity",
        "UnitPrice",
        "TotalAmount"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        st.error(f"Missing columns: {missing_columns}")
        st.stop()

    # DATA PREPARATION

    for col in [
        "Quantity",
        "UnitPrice",
        "TotalAmount",
        "Anomaly_Score"
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(subset=required_columns)

    # KPI METRICS

    total_transactions = len(df)

    anomaly_df = df[
        df["Anomaly"] == "Anomaly"
    ].copy()

    normal_df = df[
        df["Anomaly"] == "Normal"
    ].copy()

    anomaly_count = len(anomaly_df)
    normal_count = len(normal_df)

    anomaly_percentage = (
        anomaly_count / total_transactions * 100
        if total_transactions else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Normal Transactions", f"{normal_count:,}")
    col3.metric("Anomalies Detected", f"{anomaly_count:,}")
    col4.metric("Anomaly Rate", f"{anomaly_percentage:.2f}%")

    st.caption(
        "An anomaly is a statistically unusual transaction, "
        "not necessarily a fraudulent one."
    )

    st.divider()

    # ANOMALY DISTRIBUTION

    st.subheader("Transaction Pattern Analysis")

    col1, col2 = st.columns(2)

    with col1:

        distribution = pd.DataFrame({
            "Transaction Type": ["Normal", "Anomaly"],
            "Count": [normal_count, anomaly_count]
        })

        fig = px.pie(
            distribution,
            names="Transaction Type",
            values="Count",
            hole=0.65,
            color="Transaction Type",
            color_discrete_map={
                "Normal": "#34D399",
                "Anomaly": "#F87171"
            },
            template="plotly_dark",
            title="Normal vs Unusual Transactions"
        )

        fig.update_layout(
            paper_bgcolor="#0B1020",
            plot_bgcolor="#0B1020"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.histogram(
            df,
            x="Anomaly_Score",
            color="Anomaly",
            nbins=35,
            color_discrete_map={
                "Normal": "#34D399",
                "Anomaly": "#F87171"
            },
            template="plotly_dark",
            title="Anomaly Score Distribution"
        )

        fig.update_layout(
            paper_bgcolor="#0B1020",
            plot_bgcolor="#151D32"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # TRANSACTION BEHAVIOUR

    st.subheader("Transaction Behaviour")

    plot_df = df.sample(
        n=min(10000, len(df)),
        random_state=42
    )

    fig = px.scatter(
        plot_df,
        x="Quantity",
        y="TotalAmount",
        color="Anomaly",
        hover_data=[
            col for col in [
                "InvoiceNo",
                "CustomerID",
                "UnitPrice",
                "Anomaly_Score"
            ]
            if col in plot_df.columns
        ],
        color_discrete_map={
            "Normal": "#34D399",
            "Anomaly": "#F87171"
        },
        opacity=0.65,
        template="plotly_dark",
        title="Quantity vs Transaction Value"
    )

    fig.update_layout(
        paper_bgcolor="#0B1020",
        plot_bgcolor="#151D32",
        xaxis_title="Purchase Quantity",
        yaxis_title="Transaction Value (£)"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # EXPLORE UNUSUAL TRANSACTIONS

    st.subheader("Explore Unusual Transactions")

    if not anomaly_df.empty:

        min_score = float(anomaly_df["Anomaly_Score"].min())
        max_score = float(anomaly_df["Anomaly_Score"].max())

        if min_score < max_score:

            score_step = max(
                (max_score - min_score) / 1000,
                0.000001
            )

            score_range = st.slider(
                "Anomaly Score Range",
                min_value=min_score,
                max_value=max_score,
                value=(min_score, max_score),
                step=score_step,
                format="%.5f",
                key="anomaly_score_range"
            )

            filtered = anomaly_df[
                anomaly_df["Anomaly_Score"].between(
                    score_range[0],
                    score_range[1]
                )
            ].copy()

            st.caption(
                f"Selected score range: "
                f"{score_range[0]:.5f} to {score_range[1]:.5f}"
            )

        else:
            filtered = anomaly_df.copy()
            st.info("All detected anomalies have the same score.")

    else:
        filtered = anomaly_df.copy()
        st.info("No unusual transactions detected.")

    # CUSTOMER SEARCH

    search_id = st.text_input(
        "Search Customer ID",
        placeholder="Enter Customer ID...",
        key="anomaly_customer_search"
    )

    if search_id.strip() and "CustomerID" in filtered.columns:

        filtered = filtered[
            filtered["CustomerID"].astype(str).str.contains(
                search_id.strip(),
                case=False,
                na=False
            )
        ]

    filtered = filtered.sort_values(
        "Anomaly_Score",
        ascending=False
    )

    st.metric(
        "Matching Unusual Transactions",
        f"{len(filtered):,}"
    )

    display_columns = [
        col for col in [
            "InvoiceNo",
            "InvoiceDate",
            "CustomerID",
            "Description",
            "Quantity",
            "UnitPrice",
            "TotalAmount",
            "Country",
            "Anomaly_Score"
        ]
        if col in filtered.columns
    ]

    st.dataframe(
        filtered[display_columns].head(200),
        use_container_width=True,
        hide_index=True
    )

    # DOWNLOAD

    st.download_button(
        "Download Anomaly Records",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="anomaly_records.csv",
        mime="text/csv"
    )

    st.caption(
        "Higher anomaly scores indicate more unusual observations "
        "according to the fitted Isolation Forest model."
    )


# ==========================================================
# REPORTS & INSIGHTS PAGE
# ==========================================================

elif page == "Reports & Insights":

    st.title("Reports & Insights")

    st.caption(
        "Business intelligence, customer insights "
        "and downloadable analytical reports."
    )

    st.divider()

    # ------------------------------------------------------
    # BUSINESS PERFORMANCE
    # ------------------------------------------------------

    st.subheader("Business Performance")

    if retail is not None:

        total_revenue = retail["TotalAmount"].sum()

        total_customers = retail["CustomerID"].nunique()

        total_orders = retail["InvoiceNo"].nunique()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Revenue",
            f"£{total_revenue:,.2f}"
        )

        col2.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

        col3.metric(
            "Total Orders",
            f"{total_orders:,}"
        )

        st.download_button(
            "Download Retail Dataset",
            data=retail.to_csv(index=False).encode("utf-8"),
            file_name="retail_report.csv",
            mime="text/csv"
        )

    else:

        st.info("Retail dataset unavailable.")

    st.divider()

    # ------------------------------------------------------
    # CUSTOMER SEGMENTATION INSIGHTS
    # ------------------------------------------------------

    st.subheader("Customer Segmentation Insights")

    if clusters is not None:

        cluster_column = get_cluster_column(clusters)

        if cluster_column:

            segment_summary = (
                clusters.groupby(cluster_column)
                .agg(
                    Customers=(cluster_column, "size"),
                    Avg_Recency=("Recency", "mean"),
                    Avg_Frequency=("Frequency", "mean"),
                    Avg_Monetary=("Monetary", "mean")
                )
                .round(2)
                .reset_index()
            )

            st.dataframe(
                segment_summary,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Segmentation Report",
                data=clusters.to_csv(index=False).encode("utf-8"),
                file_name="segmentation_report.csv",
                mime="text/csv"
            )

    else:

        st.info("Customer segmentation data unavailable.")

    st.divider()

    # ------------------------------------------------------
    # CHURN INSIGHTS
    # ------------------------------------------------------

    st.subheader("Churn Intelligence Insights")

    if churn is not None:

        if "Risk_Category" in churn.columns:

            risk_summary = (
                churn["Risk_Category"]
                .value_counts()
                .reindex(
                    ["High Risk", "Medium Risk", "Low Risk"],
                    fill_value=0
                )
                .reset_index()
            )

            risk_summary.columns = [
                "Risk Category",
                "Prediction Records"
            ]

            st.dataframe(
                risk_summary,
                use_container_width=True,
                hide_index=True
            )

            high_risk_count = (
                churn["Risk_Category"] == "High Risk"
            ).sum()

            st.metric(
                "High-Risk Prediction Records",
                f"{high_risk_count:,}"
            )

            st.download_button(
                "Download Churn Report",
                data=churn.to_csv(index=False).encode("utf-8"),
                file_name="churn_report.csv",
                mime="text/csv"
            )

    else:

        st.info("Churn prediction data unavailable.")

    st.divider()

    # ------------------------------------------------------
    # ANOMALY INSIGHTS
    # ------------------------------------------------------

    st.subheader("Anomaly Detection Insights")

    if anomalies is not None:

        anomaly_count = (
            anomalies["Anomaly"] == "Anomaly"
        ).sum()

        normal_count = (
            anomalies["Anomaly"] == "Normal"
        ).sum()

        total = len(anomalies)

        anomaly_rate = (
            anomaly_count / total * 100
            if total else 0
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Transactions Analysed",
            f"{total:,}"
        )

        col2.metric(
            "Unusual Transactions",
            f"{anomaly_count:,}"
        )

        col3.metric(
            "Anomaly Rate",
            f"{anomaly_rate:.2f}%"
        )

        st.download_button(
            "Download Anomaly Report",
            data=anomalies.to_csv(index=False).encode("utf-8"),
            file_name="anomaly_report.csv",
            mime="text/csv"
        )

    else:

        st.info("Anomaly detection data unavailable.")

    st.divider()

    # ------------------------------------------------------
    # COMPLETE REPORT EXPORT
    # ------------------------------------------------------

    st.subheader("Export Reports")

    st.write(
        "Download individual datasets for further analysis "
        "or business reporting."
    )

    export_files = {
        "Retail Transactions": retail,
        "Customer Segmentation": clusters,
        "Churn Predictions": churn,
        "Anomaly Results": anomalies
    }

    for report_name, report_df in export_files.items():

        if report_df is not None:

            st.download_button(
                label=f"Download {report_name}",
                data=report_df.to_csv(index=False).encode("utf-8"),
                file_name=(
                    report_name.lower()
                    .replace(" ", "_") + ".csv"
                ),
                mime="text/csv",
                key=f"export_{report_name}"
            )

