
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. Load RFM table
rfm = pd.read_csv("outputs/rfm_table.csv")

# 2. Select RFM features
features = ["recency", "frequency", "monetary"]
X = rfm[features]

# 3. Standardize RFM features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Fit K-means with K = 4
model = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = model.fit_predict(X_scaled)

# 5. Calculate segment profiles
profile = rfm.groupby("segment_id").agg(
    customer_count=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")
).round(2)

print("\nCUSTOMER SEGMENT PROFILES")
print("=" * 65)
print(profile.to_string())

# 6. Business names
segment_names = {
    0: "Regular Value Customers",
    1: "Premium Customers",
    2: "Loyal Frequent Customers",
    3: "At-Risk Inactive Customers"
}

# 7. Segment recommendations
recommendations = {
    0: "Encourage repeat purchases through personalized grocery bundles and targeted offers.",
    1: "Retain high-value customers through exclusive benefits and personalized premium offers.",
    2: "Reward frequent purchases with loyalty points and subscription-style offers.",
    3: "Launch win-back campaigns with personalized discounts to encourage customers to return."
}

profile["segment_name"] = profile.index.map(segment_names)

premium_segment = profile["mean_monetary"].idxmax()

# 8. Print names and recommendations
print("\nBUSINESS SEGMENT NAMES")
print("=" * 65)

for segment_id, row in profile.iterrows():
    print(f"\nCluster {segment_id}: {row['segment_name']}")
    print("Customers:", row["customer_count"])
    print("Recommendation:", recommendations[segment_id])

print("\nPremium cluster:", premium_segment)

# 9. PCA with two components
pca = PCA(n_components=2)
pca.fit(X_scaled)

variance = pca.explained_variance_ratio_

print("\nPCA EXPLAINED VARIANCE")
print("=" * 45)
print("PC1:", round(variance[0], 3))
print("PC2:", round(variance[1], 3))
print("Total:", round(variance.sum(), 3))

# 10. Validate reference checkpoints
expected_sizes = {
    0: 436,
    1: 138,
    2: 419,
    3: 182
}

assert profile["customer_count"].to_dict() == expected_sizes

assert premium_segment == 1

assert abs(variance[0] - 0.634) <= 0.02
assert abs(variance[1] - 0.218) <= 0.02
assert abs(variance.sum() - 0.852) <= 0.02

# 11. Add business names and Premium flag
rfm["segment_name"] = rfm["segment_id"].map(segment_names)
rfm["is_premium"] = rfm["segment_id"] == premium_segment

# 12. Save required customer segments
output = rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium"
    ]
]

Path("outputs").mkdir(exist_ok=True)

output.to_csv(
    "outputs/customer_segments.csv",
    index=False
)

# Save segment profiles
profile.to_csv("outputs/segment_profiles.csv")

# 13. Final validation
assert len(output) == 1175
assert output["is_premium"].sum() == 138

assert list(output.columns) == [
    "customer_id",
    "recency",
    "frequency",
    "monetary",
    "segment_id",
    "segment_name",
    "is_premium"
]

print("\nCUSTOMER SEGMENTS SAVED")
print("Total customers:", len(output))
print("Premium customers:", output["is_premium"].sum())
print("Output: outputs/customer_segments.csv")

print("\nALL TASK 3 CHECKPOINTS PASSED")
