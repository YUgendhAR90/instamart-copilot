
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("outputs/lapse_dataset.csv")
segments = pd.read_csv("outputs/customer_segments.csv")

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


# --------------------------------------------------
# 3. PREPROCESSING PIPELINE
# --------------------------------------------------

def create_preprocessor():

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

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features)
        ],
        verbose_feature_names_out=False
    )


# --------------------------------------------------
# 4. RETRAIN BEST LOGISTIC REGRESSION
# --------------------------------------------------

print("\nLOGISTIC REGRESSION")
print("=" * 55)

logistic_pipeline = Pipeline([
    ("preprocessor", create_preprocessor()),
    ("classifier", LogisticRegression(max_iter=1000))
])

logistic_grid = GridSearchCV(
    logistic_pipeline,
    {
        "classifier__C": [0.01, 0.1, 1, 10, 100]
    },
    cv=5,
    scoring="f1",
    n_jobs=-1
)

logistic_grid.fit(X_train, y_train)

best_logistic = logistic_grid.best_estimator_

print("Best parameters:", logistic_grid.best_params_)
print("Best CV F1:", round(logistic_grid.best_score_, 4))


# --------------------------------------------------
# 5. RETRAIN BEST RANDOM FOREST
# --------------------------------------------------

print("\nRANDOM FOREST")
print("=" * 55)

forest_pipeline = Pipeline([
    ("preprocessor", create_preprocessor()),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    )
])

forest_grid = GridSearchCV(
    forest_pipeline,
    {
        "classifier__max_depth": [3, 5, 8],
        "classifier__min_samples_leaf": [1, 5, 10]
    },
    cv=5,
    scoring="f1",
    n_jobs=-1
)

forest_grid.fit(X_train, y_train)

best_forest = forest_grid.best_estimator_

print("Best parameters:", forest_grid.best_params_)
print("Best CV F1:", round(forest_grid.best_score_, 4))


# --------------------------------------------------
# 6. SELECT BEST MODEL
# --------------------------------------------------

if forest_grid.best_score_ >= logistic_grid.best_score_:
    selected_model = best_forest
    selected_name = "Random Forest"
else:
    selected_model = best_logistic
    selected_name = "Logistic Regression"

# Verify selected model meets test F1 requirement
selected_predictions = selected_model.predict(X_test)

selected_f1 = f1_score(
    y_test,
    selected_predictions,
    pos_label=1
)

assert selected_f1 >= 0.55

print("\nSELECTED MODEL:", selected_name)
print("Test F1:", round(selected_f1, 4))


# --------------------------------------------------
# 7. RANDOM FOREST FEATURE IMPORTANCE
# --------------------------------------------------

print("\nTOP FIVE RANDOM FOREST FEATURES")
print("=" * 55)

rf_preprocessor = best_forest.named_steps["preprocessor"]
rf_classifier = best_forest.named_steps["classifier"]

feature_names = rf_preprocessor.get_feature_names_out()

importances = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_classifier.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

top_five = importances.head(5)

print(top_five.to_string(index=False))


# --------------------------------------------------
# 8. LOGISTIC REGRESSION COEFFICIENTS
# --------------------------------------------------

print("\nLOGISTIC REGRESSION COEFFICIENTS")
print("=" * 55)

lr_preprocessor = best_logistic.named_steps["preprocessor"]
lr_classifier = best_logistic.named_steps["classifier"]

lr_feature_names = lr_preprocessor.get_feature_names_out()
lr_coefficients = lr_classifier.coef_[0]

coefficient_map = dict(
    zip(lr_feature_names, lr_coefficients)
)

selected_coefficients = [
    "recency",
    "frequency",
    "avg_delivery_minutes"
]

business_meanings = {
    "recency": {
        "positive": "More days since the last order are associated with greater lapse risk.",
        "negative": "More days since the last order are associated with lower lapse risk."
    },
    "frequency": {
        "positive": "More historical orders are associated with greater lapse risk.",
        "negative": "More historical orders are associated with lower lapse risk."
    },
    "avg_delivery_minutes": {
        "positive": "Longer historical delivery times are associated with greater lapse risk.",
        "negative": "Longer historical delivery times are associated with lower lapse risk."
    }
}

coefficient_rows = []

for feature in selected_coefficients:

    value = float(coefficient_map[feature])

    direction = (
        "positive" if value > 0
        else "negative" if value < 0
        else "zero"
    )

    meaning = (
        business_meanings[feature][direction]
        if direction != "zero"
        else "This feature has no directional linear effect in this fitted model."
    )

    coefficient_rows.append({
        "feature": feature,
        "coefficient": value,
        "sign": direction,
        "business_meaning": meaning
    })

coefficient_report = pd.DataFrame(coefficient_rows)

print(coefficient_report.to_string(index=False))


# --------------------------------------------------
# 9. SAVE AND RELOAD SELECTED MODEL
# --------------------------------------------------

Path("outputs").mkdir(exist_ok=True)

model_path = "outputs/lapse_model.joblib"

joblib.dump(selected_model, model_path)

# Predict probabilities before reloading
probabilities_before = selected_model.predict_proba(X_test)

# Reload saved model
loaded_model = joblib.load(model_path)

# Predict probabilities after reloading
probabilities_after = loaded_model.predict_proba(X_test)

# Verify exact equality
exact_match = np.array_equal(
    probabilities_before,
    probabilities_after
)

print("\nMODEL PERSISTENCE")
print("=" * 55)

print("Model saved:", model_path)
print("Model reloaded successfully")
print("Test probabilities exactly identical:", exact_match)

assert exact_match

print(
    "Maximum probability difference:",
    np.max(np.abs(probabilities_before - probabilities_after))
)

# Show a probability preview
preview = pd.DataFrame({
    "customer_id": df.loc[X_test.index, "customer_id"].values,
    "lapse_probability": probabilities_after[:, 1]
})

print("\nFIRST FIVE TEST PROBABILITIES")
print(preview.head(5).to_string(index=False))


# --------------------------------------------------
# 10. FIND THE HIGHEST-MONETARY SEGMENT
# --------------------------------------------------

segment_means = segments.groupby("segment_id")["monetary"].mean()

highest_segment_id = segment_means.idxmax()

premium_ids = set(
    segments.loc[
        segments["segment_id"] == highest_segment_id,
        "customer_id"
    ]
)

print("\nHIGHEST-MONETARY SEGMENT")
print("=" * 55)

print("Segment ID:", highest_segment_id)
print("Mean monetary value: INR", f"{segment_means.max():,.2f}")
print("Premium customers:", len(premium_ids))

assert highest_segment_id == 1
assert len(premium_ids) == 138


# --------------------------------------------------
# 11. CREATE PREMIUM RETENTION LIST
# --------------------------------------------------

test_results = pd.DataFrame({
    "customer_id": df.loc[X_test.index, "customer_id"].values,
    "lapse_probability": probabilities_after[:, 1],
    "actually_lapsed": y_test.to_numpy()
})

premium_test = test_results[
    test_results["customer_id"].isin(premium_ids)
].copy()

# Rank highest predicted lapse probability first
retention_list = premium_test.sort_values(
    ["lapse_probability", "customer_id"],
    ascending=[False, True],
    kind="stable"
).head(10)

# Keep exactly the required columns
retention_list = retention_list[
    [
        "customer_id",
        "lapse_probability",
        "actually_lapsed"
    ]
]

# Save list
retention_list.to_csv(
    "outputs/retention_list.csv",
    index=False
)

# Count truly lapsed customers in the selected list
truly_lapsed = int(
    retention_list["actually_lapsed"].sum()
)

print("\nPREMIUM RETENTION LIST")
print("=" * 65)

print("Premium customers in test set:", len(premium_test))
print("Customers in retention list:", len(retention_list))

print("\nTop customers:")
print(retention_list.to_string(index=False))

print("\nTruly lapsed among selected customers:", truly_lapsed)


# --------------------------------------------------
# 12. FINAL VALIDATION
# --------------------------------------------------

assert list(retention_list.columns) == [
    "customer_id",
    "lapse_probability",
    "actually_lapsed"
]

assert len(retention_list) == min(10, len(premium_test))
assert retention_list["customer_id"].isin(premium_ids).all()

# Save reports
top_five.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

coefficient_report.to_csv(
    "outputs/logistic_coefficients.csv",
    index=False
)

preview.to_csv(
    "outputs/test_probabilities.csv",
    index=False
)

print("\n" + "=" * 65)
print("ALL TASK 8 CHECKPOINTS PASSED")
print("=" * 65)

print("Model: outputs/lapse_model.joblib")
print("Retention list: outputs/retention_list.csv")
print("Feature importance: outputs/feature_importance.csv")
print("Logistic coefficients: outputs/logistic_coefficients.csv")
