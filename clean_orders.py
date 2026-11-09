
import pandas as pd
from pathlib import Path

# 1. Load raw orders and products
orders = pd.read_csv("data/raw/orders.csv")
products = pd.read_csv("data/raw/products.csv")

# 2. Convert order timestamp to datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# 3. Remove exact duplicate rows
orders = orders.drop_duplicates().copy()

# 4. Sort by timestamp and keep latest record per order_id
orders = orders.sort_values("order_ts", kind="stable")
orders = orders.drop_duplicates(
    subset="order_id",
    keep="last"
).copy()

# 5. Convert order_amount into numeric values
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"],
    errors="coerce"
)

# 6. Convert delivery_minutes into numeric values
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce"
)

# 7. Replace negative delivery times with missing values
orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = float("nan")

# 8. Join product category and price
orders = orders.merge(
    products[["product_id", "category", "price_inr"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

# 9. Calculate missing order amounts
calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

orders["order_amount"] = orders["order_amount"].fillna(
    calculated_amount
)

# 10. Fill missing delivery times only for completed orders
completed_mask = orders["status"] == "completed"

category_median = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

orders.loc[completed_mask, "delivery_minutes"] = (
    orders.loc[completed_mask, "delivery_minutes"]
    .fillna(category_median)
)

# 11. Remove temporary product columns
orders = orders.drop(columns=["category", "price_inr"])

# 12. Convert timestamp back to required text format
orders["order_ts"] = orders["order_ts"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)

# 13. Save cleaned orders
Path("data/clean").mkdir(parents=True, exist_ok=True)

orders.to_csv(
    "data/clean/orders.csv",
    index=False
)

# 14. Verify results
completed = orders[orders["status"] == "completed"]
cancelled = orders[orders["status"] == "cancelled"]

print("\nORDER CLEANING COMPLETED")

print("Total rows:", len(orders))
print("Completed orders:", len(completed))
print("Cancelled orders:", len(cancelled))
print("Pending orders:", (orders["status"] == "pending").sum())

print(
    "Missing order amounts:",
    orders["order_amount"].isna().sum()
)

print(
    "Missing delivery in completed orders:",
    completed["delivery_minutes"].isna().sum()
)

print(
    "Missing delivery in cancelled orders:",
    cancelled["delivery_minutes"].isna().sum()
)

print(
    "Completed order revenue: INR",
    f"{completed['order_amount'].sum():,.2f}"
)

print("\nOutput saved: data/clean/orders.csv")
