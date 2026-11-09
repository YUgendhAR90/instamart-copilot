
import pandas as pd

# Load raw datasets
customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")
orders = pd.read_csv("data/raw/orders.csv")


# 1. Display shape, data types and missing percentages
def audit_table(name, df):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("\nShape:", df.shape)

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values (%):")
    missing = df.isna().mean() * 100
    print(missing.round(2))


audit_table("CUSTOMERS", customers)
audit_table("PRODUCTS", products)
audit_table("ORDERS", orders)


# 2. Orders audit
numeric_amount = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

numeric_delivery = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

print("\n" + "=" * 60)
print("ORDERS AUDIT")
print("=" * 60)

print("Raw order rows:", len(orders))
print("Exact duplicate rows:", orders.duplicated().sum())
print("Pending orders:", (orders["status"] == "pending").sum())
print("Distinct order IDs:", orders["order_id"].nunique())

print(
    "Non-numeric order amounts:",
    numeric_amount.isna().sum()
)

print("Negative delivery minutes:", (numeric_delivery < 0).sum())
print("Blank delivery minutes:", orders["delivery_minutes"].isna().sum())


# 3. Customers audit
numeric_age = pd.to_numeric(customers["age"], errors="coerce")

print("\n" + "=" * 60)
print("CUSTOMERS AUDIT")
print("=" * 60)

print("Blank ages:", customers["age"].isna().sum())

print(
    "Ages outside 10-100:",
    (numeric_age.notna() & ~numeric_age.between(10, 100)).sum()
)

print("Distinct raw city spellings:", customers["city"].nunique())
print("Distinct non-blank gender spellings:", customers["gender"].nunique())
print("Blank genders:", customers["gender"].isna().sum())

print("\nRAW DATA AUDIT COMPLETED")
