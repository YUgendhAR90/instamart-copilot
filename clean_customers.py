
import pandas as pd
from pathlib import Path

# Load raw customer data
df = pd.read_csv("data/raw/customers.csv")

# 1. Standardize city names
df["city"] = df["city"].str.strip().str.title()

# 2. Standardize gender
df["gender"] = df["gender"].str.strip().str.lower()

gender_map = {
    "male": "male",
    "m": "male",
    "female": "female",
    "f": "female"
}

df["gender"] = df["gender"].map(gender_map)
df["gender"] = df["gender"].fillna("unknown")

# 3. Convert age to numeric
df["age"] = pd.to_numeric(df["age"], errors="coerce")

# Treat ages outside 10-100 as missing
df.loc[~df["age"].between(10, 100), "age"] = float("nan")

# 4. Calculate overall median as fallback
overall_median = df["age"].median()

# 5. Fill missing ages using city median
city_median = df.groupby("city")["age"].transform("median")

df["age"] = df["age"].fillna(city_median)

# Fill any remaining missing ages using overall median
df["age"] = df["age"].fillna(overall_median)

# 6. Create output folder
Path("data/clean").mkdir(parents=True, exist_ok=True)

# Save cleaned customer data
df.to_csv("data/clean/customers.csv", index=False)

# 7. Display verification results
city_order = [
    "Chennai", "Delhi", "Bengaluru",
    "Hyderabad", "Mumbai", "Pune"
]

gender_order = ["male", "female", "unknown"]

print("\nCUSTOMER CLEANING COMPLETED")

print("\nCity Counts:")
print(df["city"].value_counts().reindex(city_order))

print("\nGender Counts:")
print(df["gender"].value_counts().reindex(gender_order))

print("\nMissing Ages:", df["age"].isna().sum())
print("Minimum Age:", df["age"].min())
print("Maximum Age:", df["age"].max())

print("\nOutput saved: data/clean/customers.csv")
