
import pandas as pd
from pathlib import Path

# Fixed snapshot time from the assignment
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")

# 1. Load Part A cleaned datasets
orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")

# 2. Keep only completed orders
completed = orders[
    orders["status"] == "completed"
].copy()

# 3. Convert required columns
completed["order_ts"] = pd.to_datetime(
    completed["order_ts"],
    errors="raise"
)

completed["order_amount"] = pd.to_numeric(
    completed["order_amount"],
    errors="raise"
)

# 4. Calculate RFM values for each customer
rfm = completed.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

# 5. Calculate recency in whole days
rfm["recency"] = (
    SNAPSHOT - rfm["last_order_ts"]
).dt.days

# 6. Keep only the required RFM columns
rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
].copy()

# 7. Validate customer IDs against Part A customers
known_ids = set(customers["customer_id"])

assert rfm["customer_id"].isin(known_ids).all()
assert rfm["customer_id"].is_unique

# 8. Save RFM table
Path("outputs").mkdir(exist_ok=True)

rfm.to_csv(
    "outputs/rfm_table.csv",
    index=False
)

# 9. Display RFM statistics
print("\nRFM TABLE COMPLETED")
print("=" * 50)

print("Customers with completed orders:", len(rfm))

print("\nRecency statistics:")
print("Median:", rfm["recency"].median())
print("Maximum:", rfm["recency"].max())

print("\nFrequency statistics:")
print("Mean:", round(rfm["frequency"].mean(), 2))
print("Maximum:", rfm["frequency"].max())

print("\nMonetary statistics:")
print("Mean: INR", f"{rfm['monetary'].mean():,.2f}")
print("Maximum: INR", f"{rfm['monetary'].max():,.2f}")

# 10. Assignment reference validation
assert len(rfm) == 1175
assert rfm["recency"].median() == 19
assert rfm["recency"].max() == 180
assert round(rfm["frequency"].mean(), 2) == 9.46
assert rfm["frequency"].max() == 33
assert round(rfm["monetary"].mean(), 2) == 2552.06
assert round(rfm["monetary"].max(), 2) == 14766.55

print("\nALL RFM CHECKPOINTS PASSED")
print("Output saved: outputs/rfm_table.csv")
