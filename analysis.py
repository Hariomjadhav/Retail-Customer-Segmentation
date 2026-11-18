import pandas as pd

# Load Clustered Data
df = pd.read_csv("clustered_customers.csv", index_col=0)

features = [
    "PurchaseFrequency",
    "TotalSpending",
    "TotalQuantity",
    "Recency",
    "AverageOrderValue"
]

# K=3 Cluster Analysis
print("\n========== K=3 CUSTOMER BEHAVIOUR ==========")

k3_analysis = df.groupby("Cluster_K3")[features].mean().round(2)

print(k3_analysis)

# K=4 Cluster Analysis
print("\n========== K=4 CUSTOMER BEHAVIOUR ==========")

k4_analysis = df.groupby("Cluster_K4")[features].mean().round(2)

print(k4_analysis)

# Customer Count Comparison
print("\n========== CUSTOMER DISTRIBUTION ==========")

print("\nK=3:")
print(df["Cluster_K3"].value_counts().sort_index())

print("\nK=4:")
print(df["Cluster_K4"].value_counts().sort_index())

# Save Analysis
k3_analysis.to_csv("k3_cluster_analysis.csv")
k4_analysis.to_csv("k4_cluster_analysis.csv")

print("\nCluster Behaviour Analysis Completed!")