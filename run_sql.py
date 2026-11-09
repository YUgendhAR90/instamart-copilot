
import re
import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = "data/instamart.db"
SQL_PATH = "sql/queries.sql"

# Read SQL file
sql_text = Path(SQL_PATH).read_text(encoding="utf-8")

# Extract seven queries
parts = re.split(r"(?m)^-- Q[1-7]:.*$", sql_text)

queries = [
    part.strip().rstrip(";").strip()
    for part in parts
    if part.strip()
]

assert len(queries) == 7, f"Expected 7 queries, found {len(queries)}"

# Execute queries
with sqlite3.connect(DB_PATH) as conn:
    results = [
        pd.read_sql_query(query, conn)
        for query in queries
    ]

q1, q2, q3, q4, q5, q6, q7 = results

print("\nSQL ANALYSIS RESULTS")
print("=" * 50)

# Q1
print("\nQ1 - WHERE AND HAVING")
print("Customers:", len(q1))
assert len(q1) == 30

# Q2
print("\nQ2 - THREE TABLE JOIN")
print("Revenue groups:", len(q2))

total_revenue = round(q2["revenue"].sum(), 2)
print("Total revenue: INR", f"{total_revenue:,.2f}")

largest = q2.loc[q2["revenue"].idxmax()]
print("Largest city/category:", largest["city"], "/", largest["category"])
print("Largest revenue: INR", f"{largest['revenue']:,.2f}")

assert len(q2) == 30
assert total_revenue == 2998668.80
assert largest["city"] == "Bengaluru"
assert largest["category"] == "Electronics"
assert round(largest["revenue"], 2) == 233909.45

# Q3
print("\nQ3 - CORRELATED SUBQUERY")
print("Products above category average:", len(q3))
assert len(q3) == 19

# Q4
print("\nQ4 - LEFT JOIN")
print("Customers without completed orders:", len(q4))
assert len(q4) == 25

# Q5
print("\nQ5 - COMMON TABLE EXPRESSIONS")
print("Customers above average spending:", len(q5))

average_spend = q5["average_spend"].iloc[0]
print("Average customer spending: INR", f"{average_spend:,.2f}")

assert len(q5) == 550
assert round(average_spend, 2) == 2552.06

# Q6
print("\nQ6 - WINDOW FUNCTIONS")

rank_differences = (
    q6["rank_position"] != q6["row_number_position"]
).sum()

print("Customers ranked:", len(q6))
print("RANK and ROW_NUMBER differences:", rank_differences)

top_customers = q6[q6["row_number_position"] == 1]

print("\nTop customer in each city:")
print(
    top_customers[
        ["city", "customer_id", "total_spend"]
    ].to_string(index=False)
)

expected_top = {
    "Bengaluru": ("C0082", 12856.20),
    "Chennai": ("C0930", 10151.40),
    "Delhi": ("C0291", 13742.60),
    "Hyderabad": ("C0743", 12730.05),
    "Mumbai": ("C0693", 14766.55),
    "Pune": ("C0426", 10377.10)
}

for city, (customer, spend) in expected_top.items():
    row = top_customers[top_customers["city"] == city].iloc[0]
    assert row["customer_id"] == customer
    assert round(row["total_spend"], 2) == spend

assert rank_differences == 8

# Q7
print("\nQ7 - RUNNING TOTAL")

print("Daily revenue rows:", len(q7))

final_revenue = q7["cumulative_revenue"].iloc[-1]

print("Final cumulative revenue: INR", f"{final_revenue:,.2f}")

assert len(q7) == 183
assert round(final_revenue, 2) == 2998668.80

print("\n" + "=" * 50)
print("ALL 7 SQL QUERIES PASSED")
print("=" * 50)
