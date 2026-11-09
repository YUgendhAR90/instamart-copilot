
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("outputs/lapse_dataset.csv")

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

# --------------------------------------------------
# 2. RECREATE ORIGINAL TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nORIGINAL TRAINING DISTRIBUTION")
print("=" * 50)
print(y_train.value_counts().sort_index())

assert y_train.value_counts().to_dict() == {
    0: 570,
    1: 354
}

# --------------------------------------------------
# 3. CREATE PREPROCESSING PIPELINE
# --------------------------------------------------

def create_pipeline():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=0.1,
            max_iter=1000
        ))
    ])


# --------------------------------------------------
# 4. TRAIN ORIGINAL LOGISTIC REGRESSION
# --------------------------------------------------

original_model = create_pipeline()

original_model.fit(X_train, y_train)

original_predictions = original_model.predict(X_test)


# --------------------------------------------------
# 5. CREATE BALANCED TRAINING DATA
# --------------------------------------------------

train_data = X_train.reset_index(drop=True).copy()
train_data["lapsed"] = y_train.reset_index(drop=True)

class_counts = train_data["lapsed"].value_counts()

majority_class = class_counts.idxmax()
minority_class = class_counts.idxmin()

majority_count = int(class_counts[majority_class])
minority_count = int(class_counts[minority_class])

duplicates_needed = majority_count - minority_count

# Select minority rows randomly with replacement
minority_rows = train_data[
    train_data["lapsed"] == minority_class
]

duplicates = minority_rows.sample(
    n=duplicates_needed,
    replace=True,
    random_state=42
)

# Keep every original row and add duplicates
balanced_data = pd.concat(
    [train_data, duplicates],
    ignore_index=True
)

# Shuffle balanced training data
balanced_data = balanced_data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

X_balanced = balanced_data[feature_columns]
y_balanced = balanced_data["lapsed"]

print("\nBALANCED TRAINING DISTRIBUTION")
print("=" * 50)
print(y_balanced.value_counts().sort_index())

assert len(balanced_data) == 1140
assert y_balanced.value_counts().to_dict() == {
    0: 570,
    1: 570
}


# --------------------------------------------------
# 6. TRAIN BALANCED LOGISTIC REGRESSION
# --------------------------------------------------

balanced_model = create_pipeline()

balanced_model.fit(X_balanced, y_balanced)

# Same original test set, unchanged
balanced_predictions = balanced_model.predict(X_test)


# --------------------------------------------------
# 7. EVALUATE BOTH MODELS
# --------------------------------------------------

def calculate_metrics(y_true, predictions):

    return {
        "precision": precision_score(
            y_true, predictions, pos_label=1
        ),
        "recall": recall_score(
            y_true, predictions, pos_label=1
        ),
        "f1": f1_score(
            y_true, predictions, pos_label=1
        )
    }


original_metrics = calculate_metrics(
    y_test, original_predictions
)

balanced_metrics = calculate_metrics(
    y_test, balanced_predictions
)

comparison = pd.DataFrame(
    [
        {
            "model": "Original Logistic Regression",
            **original_metrics
        },
        {
            "model": "Balanced Logistic Regression",
            **balanced_metrics
        }
    ]
)

print("\nCLASS-BALANCE EXPERIMENT")
print("=" * 65)
print(comparison.to_string(index=False, float_format="%.4f"))

# --------------------------------------------------
# 8. CALCULATE METRIC CHANGES
# --------------------------------------------------

print("\nMETRIC CHANGES")
print("=" * 50)

for metric in ["precision", "recall", "f1"]:

    change = (
        balanced_metrics[metric]
        - original_metrics[metric]
    )

    print(
        metric.capitalize(),
        "change:",
        f"{change:+.4f}"
    )

print("\nORIGINAL TEST SET")
print("Test rows:", len(X_test))
print("Test lapsed customers:", int(y_test.sum()))

# --------------------------------------------------
# 9. SAVE RESULTS
# --------------------------------------------------

Path("outputs").mkdir(exist_ok=True)

comparison.to_csv(
    "outputs/balance_experiment.csv",
    index=False
)

assert len(X_test) == 232

print("\nALL TASK 7 CHECKPOINTS PASSED")
print("Output: outputs/balance_experiment.csv")
