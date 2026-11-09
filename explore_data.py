
import sqlite3
import pandas as pd

DB_PATH = "data/instamart.db"

# Connect to SQLite database
with sqlite3.connect(DB_PATH) as conn:

    # Load completed orders
    orders = pd.read_sql_query(
        "SELECT * FROM orders WHERE status = 'completed'",
        conn
    )

    customers = pd.read_sql_query(
        "SELECT customer_id, city FROM customers",
        conn
    )

    products = pd.read_sql_query(
        "SELECT product_id, category FROM products",
        conn
    )

    # Get SQL revenue for comparison
    sql_total = pd.read_sql_query(
        "SELECT SUM(order_amount) AS revenue "
        "FROM orders WHERE status = 'completed'",
        conn
    )["revenue"].iloc[0]


# Join city and category information
df = (
    orders
    .merge(customers, on="customer_id")
    .merge(products, on="product_id")
)

print("\nPART A - TASK 7")
print("=" * 60)


# --------------------------------------------------
# A. PANDAS CROSS-CHECK
# --------------------------------------------------

print("\n1. PANDAS PIVOT TABLE")

pivot = pd.pivot_table(
    df,
    index="city",
    columns="category",
    values="order_amount",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="Grand Total"
)

print(pivot.round(2))

pandas_total = pivot.loc["Grand Total", "Grand Total"]

print("\nPandas Grand Total: INR", f"{pandas_total:,.2f}")
print("SQL Total: INR", f"{sql_total:,.2f}")

assert round(pandas_total, 2) == round(sql_total, 2)


# Groupby with named aggregation
print("\n2. CITY-WISE GROUPBY REPORT")

city_report = df.groupby("city").agg(
    order_amount_sum=("order_amount", "sum"),
    order_amount_mean=("order_amount", "mean"),
    order_count=("order_amount", "count")
)

print(city_report.round(2))

# Verify reference figures
assert round(city_report.loc["Bengaluru", "order_amount_sum"], 2) == 523643.40
assert round(city_report.loc["Bengaluru", "order_amount_mean"], 2) == 276.47
assert city_report.loc["Bengaluru", "order_count"] == 1894

assert round(city_report.loc["Mumbai", "order_amount_sum"], 2) == 479058.50
assert round(city_report.loc["Mumbai", "order_amount_mean"], 2) == 297.00
assert city_report.loc["Mumbai", "order_count"] == 1613


# --------------------------------------------------
# B. EXPLORATORY DATA ANALYSIS
# --------------------------------------------------

print("\n3. ORDER AMOUNT STATISTICS")

amount = df["order_amount"].dropna()

mean = amount.mean()
median = amount.median()
skewness = amount.skew()

q1 = amount.quantile(0.25)
q3 = amount.quantile(0.75)

iqr = q3 - q1

lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

below = (amount < lower_fence).sum()
above = (amount > upper_fence).sum()

print("Mean:", round(mean, 2))
print("Median:", round(median, 2))
print("Skewness:", round(skewness, 2))
print("Q1:", round(q1, 2))
print("Q3:", round(q3, 2))
print("IQR:", round(iqr, 2))

print("Lower fence:", round(lower_fence, 2))
print("Upper fence:", round(upper_fence, 2))

print("Outliers below lower fence:", below)
print("Outliers above upper fence:", above)

assert round(mean, 2) == 269.91
assert round(median, 2) == 135.00
assert round(skewness, 2) == 4.61
assert round(q1, 2) == 64.80
assert round(q3, 2) == 297.00
assert round(upper_fence, 2) == 645.30
assert below == 0
assert above == 897


# --------------------------------------------------
# C. CORRELATION MATRIX
# --------------------------------------------------

print("\n4. CORRELATION MATRIX")

columns = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes"
]

correlation = df[columns].corr()

print(correlation.round(3))

# Find the pair with the largest absolute correlation
pairs = []

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        first = columns[i]
        second = columns[j]

        value = correlation.loc[first, second]

        pairs.append((first, second, value))

largest_pair = max(pairs, key=lambda x: abs(x[2]))

print("\nLargest absolute correlation:")
print(largest_pair[0], "and", largest_pair[1])
print("Correlation:", round(largest_pair[2], 3))


# --------------------------------------------------
# FINAL VALIDATION
# --------------------------------------------------

assert largest_pair[0] == "order_amount"
assert largest_pair[1] == "discount_pct"

print("\n" + "=" * 60)
print("ALL TASK 7 CHECKPOINTS PASSED")
print("=" * 60)


# --------------------------------------------------
# WRITTEN EXPLORATORY SUMMARY (UNDER 200 WORDS)
# --------------------------------------------------

summary = """
This synthetic dataset describes Instamart completed orders across
six cities, including products, order amounts, discounts, quantities
and delivery times.

The order amount distribution is strongly right-skewed. The mean is
INR 269.91, compared with a median of INR 135.00, and skewness is
approximately 4.61.

There are 897 high-value outliers above the upper fence of
INR 645.30 and no outliers below the lower fence. These values
are unusual but are not necessarily errors.

The strongest absolute correlation is between order amount and
discount percentage, approximately -0.095. Order amount and quantity
have a slightly smaller correlation of +0.087. These relationships
are weak, and correlation of this size does not establish causation.
"""

print("\nEXPLORATORY SUMMARY")
print(summary)
