import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Retail Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)


# ==========================================
# 2. LOAD CUSTOMER DATA
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv("customer_data.csv", index_col=0)


df = load_data()

features = [
    "PurchaseFrequency",
    "TotalSpending",
    "TotalQuantity",
    "Recency",
    "AverageOrderValue"
]


# ==========================================
# 3. DATA PREPROCESSING
# ==========================================

X = df[features].clip(lower=0)

X_log = np.log1p(X)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_log)


# ==========================================
# 4. K-MEANS CLUSTERING
# ==========================================

@st.cache_data
def perform_clustering(data):

    model_results = {}

    for k in [3, 4]:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(data)

        score = silhouette_score(data, labels)

        model_results[k] = {
            "labels": labels,
            "score": score
        }

    return model_results


results = perform_clustering(X_scaled)


# ==========================================
# 5. TITLE
# ==========================================

st.title("🛍️ Retail Customer Segmentation")

st.subheader(
    "Machine Learning Based Customer Purchasing Behaviour Analysis"
)

st.markdown("---")


# ==========================================
# 6. SIDEBAR
# ==========================================

st.sidebar.title("⚙️ Dashboard Controls")

selected_k = st.sidebar.selectbox(
    "Select Number of Clusters",
    [3, 4]
)


# ==========================================
# 7. SELECTED CLUSTER DATA
# ==========================================

selected_df = df.copy()

selected_df["Cluster"] = results[selected_k]["labels"]


# ==========================================
# 8. CLUSTER PROFILE
# ==========================================

cluster_profile = (
    selected_df.groupby("Cluster")[features]
    .mean()
)


# ==========================================
# 9. AUTOMATIC BUSINESS SEGMENT NAMES
# ==========================================

# Rank clusters using their actual characteristics.

frequency_rank = cluster_profile["PurchaseFrequency"].rank(
    ascending=False,
    method="min"
)

spending_rank = cluster_profile["TotalSpending"].rank(
    ascending=False,
    method="min"
)

recency_rank = cluster_profile["Recency"].rank(
    ascending=False,
    method="min"
)

def assign_segment(cluster):

    high_frequency = frequency_rank[cluster] == 1

    high_spending = spending_rank[cluster] == 1

    high_recency = recency_rank[cluster] == 1

    if high_frequency and high_spending:
        return "High-Value Loyal Customers"

    elif high_spending:
        return "High-Spending Customers"

    elif high_frequency:
        return "Frequent Buyers"

    elif high_recency:
        return "Inactive / At-Risk Customers"

    else:
        return "Regular Customers"


selected_df["CustomerSegment"] = (
    selected_df["Cluster"].apply(assign_segment)
)


# ==========================================
# 10. BUSINESS RECOMMENDATIONS
# ==========================================

recommendations = {

    "High-Value Loyal Customers":
        "Provide VIP membership, exclusive offers, premium services and loyalty rewards.",

    "High-Spending Customers":
        "Provide personalized product recommendations and encourage repeat purchases.",

    "Frequent Buyers":
        "Offer reward points, membership benefits and special repeat-purchase offers.",

    "Inactive / At-Risk Customers":
        "Launch comeback campaigns, personalized reminders and limited-time discounts.",

    "Regular Customers":
        "Use product bundles, introductory offers and targeted promotions."
}


# ==========================================
# 11. TOP BUSINESS METRICS
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Total Customers",
        f"{len(df):,}"
    )

with col2:

    st.metric(
        "Total Revenue",
        f"£{df['TotalSpending'].sum():,.2f}"
    )

with col3:

    st.metric(
        "Selected Clusters",
        selected_k
    )


st.markdown("---")


# ==========================================
# 12. CLUSTERING RESULTS
# ==========================================

st.header(
    f"📊 Customer Segmentation Results (K={selected_k})"
)

st.metric(
    "Silhouette Score",
    round(results[selected_k]["score"], 4)
)


# ==========================================
# 13. CUSTOMER DISTRIBUTION
# ==========================================

st.subheader("Customer Distribution")

cluster_counts = (
    selected_df["Cluster"]
    .value_counts()
    .sort_index()
    .reset_index()
)

cluster_counts.columns = [
    "Cluster",
    "Customers"
]

fig1 = px.bar(
    cluster_counts,
    x="Cluster",
    y="Customers",
    color="Cluster",
    text="Customers",
    title="Number of Customers in Each Cluster"
)

st.plotly_chart(fig1, width="stretch")


# ==========================================
# 14. CLUSTER PURCHASING BEHAVIOUR
# ==========================================

st.header("🛒 Cluster Purchasing Behaviour")

summary = (
    selected_df.groupby("Cluster")[features]
    .mean()
    .round(2)
)

st.dataframe(
    summary,
    width="stretch"
)


# ==========================================
# 15. AUTOMATIC SEGMENT ANALYSIS
# ==========================================

st.markdown("---")

st.header("🏷️ Automatic Customer Segment Analysis")

segment_summary = (
    selected_df.groupby("Cluster")
    .agg(
        Customers=("CustomerSegment", "size"),
        Segment=("CustomerSegment", "first"),
        TotalSpending=("TotalSpending", "mean"),
        PurchaseFrequency=("PurchaseFrequency", "mean"),
        Recency=("Recency", "mean"),
        AverageOrderValue=("AverageOrderValue", "mean")
    )
    .round(2)
)

st.dataframe(
    segment_summary,
    width="stretch"
)


# ==========================================
# 16. BUSINESS RECOMMENDATIONS
# ==========================================

st.header("💡 Business Recommendations")

for cluster, row in segment_summary.iterrows():

    segment = row["Segment"]

    with st.expander(
        f"Cluster {cluster}: {segment}"
    ):

        st.write(
            f"**Total Customers:** {int(row['Customers'])}"
        )

        st.write(
            f"**Average Spending:** £{row['TotalSpending']:,.2f}"
        )

        st.write(
            f"**Purchase Frequency:** {row['PurchaseFrequency']:.2f}"
        )

        st.write(
            f"**Average Recency:** {row['Recency']:.2f} days"
        )

        st.info(
            recommendations[segment]
        )


# ==========================================
# 17. SPENDING ANALYSIS
# ==========================================

st.markdown("---")

st.header("💰 Customer Spending Analysis")

fig2 = px.box(
    selected_df,
    x="Cluster",
    y="TotalSpending",
    color="Cluster",
    title="Customer Spending Distribution"
)

st.plotly_chart(fig2, width="stretch")


# ==========================================
# 18. PURCHASE FREQUENCY ANALYSIS
# ==========================================

st.header("📈 Purchase Frequency Analysis")

fig3 = px.box(
    selected_df,
    x="Cluster",
    y="PurchaseFrequency",
    color="Cluster",
    title="Purchase Frequency Across Clusters"
)

st.plotly_chart(fig3, width="stretch")


# ==========================================
# 19. K=3 VS K=4 COMPARISON
# ==========================================

st.markdown("---")

st.header("⚖️ K=3 vs K=4 Model Comparison")

comparison = pd.DataFrame({

    "Model": ["K=3", "K=4"],

    "Silhouette Score": [
        results[3]["score"],
        results[4]["score"]
    ],

    "Number of Clusters": [3, 4]

})

st.dataframe(
    comparison,
    width="stretch"
)

fig4 = px.bar(
    comparison,
    x="Model",
    y="Silhouette Score",
    color="Model",
    text="Silhouette Score",
    title="Clustering Quality Comparison"
)

st.plotly_chart(fig4, width="stretch")


# ==========================================
# 20. PROJECT CONCLUSION
# ==========================================

st.header("📋 Project Conclusion")

if results[3]["score"] > results[4]["score"]:

    st.success(
        "K=3 has a higher Silhouette Score and provides "
        "better cluster separation in this experiment."
    )

else:

    st.success(
        "K=4 has a higher Silhouette Score and provides "
        "better cluster separation in this experiment."
    )

st.info(
    "K=3 creates three customer groups, while K=4 creates "
    "four groups for more detailed customer analysis. "
    "The final choice also depends on business requirements."
)


# ==========================================
# 21. CUSTOMER DETAILS
# ==========================================

st.markdown("---")

st.header("👥 Customer Details")

st.dataframe(
    selected_df[
        features + ["Cluster", "CustomerSegment"]
    ],
    width="stretch"
)


# ==========================================
# 22. DOWNLOAD CUSTOMER REPORT
# ==========================================

st.header("📥 Download Customer Segmentation Report")

csv_data = selected_df.to_csv().encode("utf-8")

st.download_button(
    label="Download Customer Report CSV",
    data=csv_data,
    file_name=f"customer_segmentation_K{selected_k}.csv",
    mime="text/csv"
)


# ==========================================
# 23. FOOTER
# ==========================================

st.markdown("---")

st.success(
    "Retail Customer Segmentation Dashboard is Running Successfully!"
)

st.caption(
    "Algorithm: K-Means | Scaling: StandardScaler | "
    "Evaluation: Silhouette Score"
)
# ==============================
# ELBOW METHOD
# ==============================

st.header("Elbow Method - Optimal Number of Clusters")

inertia_values = []
k_values = range(1, 11)

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)
    inertia_values.append(model.inertia_)

elbow_df = pd.DataFrame({
    "Number of Clusters (K)": list(k_values),
    "Inertia (WCSS)": inertia_values
})

fig_elbow = px.line(
    elbow_df,
    x="Number of Clusters (K)",
    y="Inertia (WCSS)",
    markers=True,
    title="Elbow Method for Optimal K"
)

st.plotly_chart(fig_elbow, width="stretch")

st.write("""
**Interpretation:**

The Elbow Method helps identify a suitable number of clusters.
As the number of clusters increases, the inertia decreases.
The point where the decrease starts slowing down is called the Elbow Point.
""")

st.dataframe(elbow_df, width="stretch")