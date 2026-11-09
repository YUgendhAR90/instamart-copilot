
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Fixed assignment dates
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END = pd.Timestamp("2025-10-01 00:00:00")

# 1. Load cleaned datasets
orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"]
)

# 2. Use only historical completed orders
history = orders[
    (orders["status"] == "completed") &
    (orders["order_ts"] < CUTOFF)
].copy()

# 3. Build historical customer features
historical_features = history.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    avg_delivery_minutes=("delivery_minutes", "mean")
).reset_index()

# 4. Calculate recency
historical_features["recency"] = (
    CUTOFF - historical_features["last_order_ts"]
).dt.days

# 5. Find most frequently ordered product category
history_products = history.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

category_counts = (
    history_products.groupby(
        ["customer_id", "category"]
    )
    .size()
    .reset_index(name="order_count")
)

category_counts = category_counts.sort_values(
    ["customer_id", "order_count", "category"],
    ascending=[True, False, True]
)

favourite_category = (
    category_counts.drop_duplicates("customer_id")
    [["customer_id", "category"]]
    .rename(columns={"category": "favourite_category"})
)

# 6. Build cohort
cohort = customers[
    customers["customer_id"].isin(history["customer_id"])
].copy()

cohort = cohort.merge(
    historical_features.drop(columns="last_order_ts"),
    on="customer_id",
    how="left",
    validate="one_to_one"
)

cohort = cohort.merge(
    favourite_category,
    on="customer_id",
    how="left",
    validate="one_to_one"
)

# 7. Calculate tenure
cohort["tenure_days"] = (
    CUTOFF - cohort["signup_date"]
).dt.days

# 8. Create label using September completed orders only
september_orders = orders[
    (orders["status"] == "completed") &
    (orders["order_ts"] >= CUTOFF) &
    (orders["order_ts"] < LABEL_END)
]

active_ids = set(september_orders["customer_id"])

cohort["lapsed"] = (
    ~cohort["customer_id"].isin(active_ids)
).astype(int)

# 9. Select required features and target
dataset = cohort[
    [
        "customer_id",
        "recency",
        "frequency",
        "avg_delivery_minutes",
        "favourite_category",
        "city",
        "age",
        "gender",
        "tenure_days",
        "lapsed"
    ]
].copy()

dataset = dataset.sort_values(
    "customer_id"
).reset_index(drop=True)

# 10. Validate no missing feature values
feature_columns = [
    "recency",
    "frequency",
    "avg_delivery_minutes",
    "favourite_category",
    "city",
    "age",
    "gender",
    "tenure_days"
]

assert dataset[feature_columns].notna().all().all()
assert dataset["customer_id"].is_unique
assert dataset["lapsed"].isin([0, 1]).all()

# 11. Validate historical feature cutoff
assert history["order_ts"].max() < CUTOFF

# 12. Validate reference results
assert len(dataset) == 1156
assert dataset["lapsed"].sum() == 443
assert round(dataset["lapsed"].mean() * 100, 2) == 38.32

# 13. Stratified train-test split
X = dataset[feature_columns]
y = dataset["lapsed"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

assert len(X_train) == 924
assert len(X_test) == 232

# 14. Save dataset
Path("outputs").mkdir(exist_ok=True)

dataset.to_csv(
    "outputs/lapse_dataset.csv",
    index=False
)

# 15. Display verification
print("\nLAPSE COHORT COMPLETED")
print("=" * 55)

print("Total cohort customers:", len(dataset))
print("Lapsed customers:", dataset["lapsed"].sum())
print("Lapse percentage:", f"{dataset['lapsed'].mean() * 100:.2f}%")

print("\nFeature columns:")
print(feature_columns)

print("\nMissing feature values:")
print(dataset[feature_columns].isna().sum())

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nALL TASK 4 CHECKPOINTS PASSED")
print("Output: outputs/lapse_dataset.csv")
