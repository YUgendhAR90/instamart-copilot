
import sqlite3
import pandas as pd
from pathlib import Path

# File locations
Path("data/clean").mkdir(parents=True, exist_ok=True)

DB_PATH = "data/instamart.db"

# 1. Load cleaned datasets
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv(
    "data/clean/orders.csv",
    dtype={"order_ts": str}
)

# Store order timestamps as text
orders["order_ts"] = orders["order_ts"].astype(str)

# 2. Create SQLite database
with sqlite3.connect(DB_PATH) as conn:

    # Replace existing tables when rerunning
    for table in ["customers", "products", "orders"]:
        conn.execute(f"DROP TABLE IF EXISTS {table}")

    # Insert all three datasets
    customers.to_sql("customers", conn, index=False)
    products.to_sql("products", conn, index=False)
    orders.to_sql(
        "orders",
        conn,
        index=False,
        dtype={"order_ts": "TEXT"}
    )

    # 3. Read data back from SQLite
    db_customers = pd.read_sql_query(
        "SELECT * FROM customers", conn
    )

    db_products = pd.read_sql_query(
        "SELECT * FROM products", conn
    )

    db_orders = pd.read_sql_query(
        "SELECT * FROM orders", conn
    )

    # 4. Validate customer checkpoints
    expected_cities = {
        "Chennai": 191,
        "Delhi": 210,
        "Bengaluru": 197,
        "Hyderabad": 191,
        "Mumbai": 184,
        "Pune": 227
    }

    expected_genders = {
        "male": 560,
        "female": 610,
        "unknown": 30
    }

    city_counts = db_customers["city"].value_counts().to_dict()
    gender_counts = db_customers["gender"].value_counts().to_dict()

    assert len(db_customers) == 1200
    assert city_counts == expected_cities
    assert gender_counts == expected_genders
    assert db_customers["age"].isna().sum() == 0
    assert db_customers["age"].min() == 18
    assert db_customers["age"].max() == 60

    # 5. Validate products
    assert len(db_products) == 40

    # 6. Validate orders
    completed = db_orders[
        db_orders["status"] == "completed"
    ]

    cancelled = db_orders[
        db_orders["status"] == "cancelled"
    ]

    assert len(db_orders) == 12309
    assert len(completed) == 11110
    assert len(cancelled) == 1199
    assert (db_orders["status"] == "pending").sum() == 0

    assert db_orders["order_amount"].isna().sum() == 0

    assert completed["delivery_minutes"].isna().sum() == 0
    assert cancelled["delivery_minutes"].isna().sum() == 1199

    revenue = round(completed["order_amount"].sum(), 2)
    assert revenue == 2998668.80

    # 7. Verify SQLite timestamp storage type
    schema = pd.read_sql_query(
        "PRAGMA table_info(orders)", conn
    )

    timestamp_type = schema.loc[
        schema["name"] == "order_ts", "type"
    ].iloc[0]

    assert timestamp_type.upper() == "TEXT"

    # 8. Verify required tables
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'",
        conn
    )["name"].tolist()

    assert all(
        name in tables
        for name in ["customers", "products", "orders"]
    )

print("Database created:", DB_PATH)
print("Customers:", len(db_customers))
print("Products:", len(db_products))
print("Orders:", len(db_orders))
print("Completed revenue: INR", f"{revenue:,.2f}")
print("ALL CHECKPOINTS PASSED")
