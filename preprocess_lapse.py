
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load lapse dataset
df = pd.read_csv("outputs/lapse_dataset.csv")

# 2. Define features and target
numeric_features = [
    "recency",
    "frequency",
    "avg_delivery_minutes",
    "age",
    "tenure_days"
]

categorical_features = [
    "favourite_category",
    "city",
    "gender"
]

feature_columns = numeric_features + categorical_features

X = df[feature_columns]
y = df["lapsed"]

# 3. Split data before fitting any preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTRAIN-TEST SPLIT")
print("=" * 50)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# 4. Check missing values
print("\nMISSING VALUE CHECK")
print(X.isna().sum())

# 5. Numerical preprocessing
numeric_steps = []

if X_train[numeric_features].isna().any().any():
    numeric_steps.append(
        ("imputer", SimpleImputer(strategy="median"))
    )

numeric_steps.append(
    ("scaler", StandardScaler())
)

numeric_pipeline = Pipeline(numeric_steps)

# 6. Categorical preprocessing
categorical_steps = []

if X_train[categorical_features].isna().any().any():
    categorical_steps.append(
        ("imputer", SimpleImputer(strategy="most_frequent"))
    )

categorical_steps.append(
    (
        "onehot",
        OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False
        )
    )
)

categorical_pipeline = Pipeline(categorical_steps)

# 7. Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ],
    verbose_feature_names_out=False
)

# 8. Fit ONLY on training data
X_train_processed = preprocessor.fit_transform(X_train)

# 9. Transform test data without fitting again
X_test_processed = preprocessor.transform(X_test)

# 10. Verify scaler's learned frequency mean
fitted_scaler = (
    preprocessor
    .named_transformers_["numeric"]
    .named_steps["scaler"]
)

train_frequency_mean = fitted_scaler.mean_[
    numeric_features.index("frequency")
]

full_cohort_frequency_mean = df["frequency"].mean()

print("\nFREQUENCY MEAN COMPARISON")
print("=" * 50)

print(
    "Training scaler frequency mean:",
    round(train_frequency_mean, 4)
)

print(
    "Full cohort frequency mean:",
    round(full_cohort_frequency_mean, 4)
)

assert not np.isclose(
    train_frequency_mean,
    full_cohort_frequency_mean
), "Expected different training and full-cohort means"

# 11. Display processed dataset information
feature_names = preprocessor.get_feature_names_out()

print("\nPREPROCESSING RESULTS")
print("=" * 50)

print("Training processed shape:", X_train_processed.shape)
print("Testing processed shape:", X_test_processed.shape)

print("\nProcessed feature names:")
print(list(feature_names))

print("\nOriginal categorical columns:", len(categorical_features))
print("One-hot encoding: drop first category")

# 12. Verify preprocessing
assert len(X_train_processed) == 924
assert len(X_test_processed) == 232

assert X_train_processed.shape[1] == len(feature_names)
assert X_test_processed.shape[1] == len(feature_names)

assert len(X_train_processed) > 0
assert len(X_test_processed) > 0

print("\nALL TASK 5 CHECKPOINTS PASSED")
