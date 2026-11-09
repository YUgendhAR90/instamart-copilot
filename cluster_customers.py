
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load RFM data
rfm = pd.read_csv("outputs/rfm_table.csv")

# 2. Select RFM features
features = ["recency", "frequency", "monetary"]
X = rfm[features]

# 3. Standardize RFM features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Test K from 2 to 8
results = []

print("\nK-MEANS CLUSTERING RESULTS")
print("=" * 55)
print(f"{'K':<10}{'Inertia':<20}{'Silhouette':<15}")

for k in range(2, 9):

    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, labels)

    results.append({
        "K": k,
        "Inertia": inertia,
        "Silhouette": silhouette
    })

    print(f"{k:<10}{inertia:<20.1f}{silhouette:<15.3f}")

# 5. Save clustering metrics
metrics = pd.DataFrame(results)

metrics.to_csv(
    "outputs/kmeans_metrics.csv",
    index=False
)

# 6. Select K based on silhouette score
best_k = int(
    metrics.loc[metrics["Silhouette"].idxmax(), "K"]
)

print("\nBest K by silhouette:", best_k)

# 7. Validate assignment reference results
expected = {
    2: (1820.8, 0.465),
    3: (1268.2, 0.445),
    4: (743.1, 0.526),
    5: (617.4, 0.481),
    6: (503.2, 0.482),
    7: (408.5, 0.416),
    8: (366.5, 0.412)
}

for k, (expected_inertia, expected_silhouette) in expected.items():

    row = metrics[metrics["K"] == k].iloc[0]

    assert abs(row["Inertia"] - expected_inertia) <= 0.05 * expected_inertia, (
        f"Inertia mismatch for K={k}"
    )

    assert abs(row["Silhouette"] - expected_silhouette) <= 0.05, (
        f"Silhouette mismatch for K={k}"
    )

assert best_k == 4

print("\nSELECTED K:", best_k)
print("ALL K-MEANS CHECKPOINTS PASSED")
print("Output saved: outputs/kmeans_metrics.csv")
