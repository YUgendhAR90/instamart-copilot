
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
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
# 2. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nDATA SPLIT")
print("=" * 55)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

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
# 4. LOGISTIC REGRESSION GRID SEARCH
# --------------------------------------------------

print("\nLOGISTIC REGRESSION GRID SEARCH")
print("=" * 55)

logistic_pipeline = Pipeline([
    ("preprocessor", create_preprocessor()),
    ("classifier", LogisticRegression(max_iter=1000))
])

logistic_grid = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid={
        "classifier__C": [0.01, 0.1, 1, 10, 100]
    },
    cv=5,
    scoring="f1",
    n_jobs=-1,
    refit=True
)

# Uses training data only
logistic_grid.fit(X_train, y_train)

print("Best parameters:", logistic_grid.best_params_)
print("Best CV F1:", round(logistic_grid.best_score_, 4))


# --------------------------------------------------
# 5. RANDOM FOREST GRID SEARCH
# --------------------------------------------------

print("\nRANDOM FOREST GRID SEARCH")
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
    estimator=forest_pipeline,
    param_grid={
        "classifier__max_depth": [3, 5, 8],
        "classifier__min_samples_leaf": [1, 5, 10]
    },
    cv=5,
    scoring="f1",
    n_jobs=-1,
    refit=True
)

# Uses training data only
forest_grid.fit(X_train, y_train)

print("Number of parameter combinations:", 9)
print("Total CV fits:", 9 * 5)
print("Best parameters:", forest_grid.best_params_)
print("Best CV F1:", round(forest_grid.best_score_, 4))


# --------------------------------------------------
# 6. BASELINE MODEL
# --------------------------------------------------

print("\nMAJORITY-CLASS BASELINE")
print("=" * 55)

baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(X_train, y_train)

# Test evaluation happens only after both searches finish
baseline_predictions = baseline.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

baseline_f1 = f1_score(
    y_test,
    baseline_predictions,
    zero_division=0
)

print("Test accuracy:", round(baseline_accuracy, 4))
print("Lapsed-class F1:", round(baseline_f1, 4))


# --------------------------------------------------
# 7. EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_model(name, model):

    predictions = model.predict(X_test)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    precision = precision_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nLapsed Class (1) Metrics:")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("Accuracy:", round(accuracy, 4))

    return {
        "model": name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "confusion_matrix": matrix.tolist()
    }


# --------------------------------------------------
# 8. EVALUATE BOTH TUNED MODELS ON TEST SET
# --------------------------------------------------

# Tuning has finished. The test set is accessed here.
logistic_results = evaluate_model(
    "LOGISTIC REGRESSION",
    logistic_grid.best_estimator_
)

forest_results = evaluate_model(
    "RANDOM FOREST",
    forest_grid.best_estimator_
)


# --------------------------------------------------
# 9. SELECT BEST MODEL
# --------------------------------------------------

if forest_results["f1"] >= logistic_results["f1"]:
    selected_name = "Random Forest"
    selected_results = forest_results
    selected_model = forest_grid.best_estimator_
else:
    selected_name = "Logistic Regression"
    selected_results = logistic_results
    selected_model = logistic_grid.best_estimator_

print("\nSELECTED MODEL")
print("=" * 55)
print("Model:", selected_name)
print("Test F1:", round(selected_results["f1"], 4))

assert selected_results["f1"] >= 0.55, (
    "Selected model must achieve test F1 >= 0.55"
)


# --------------------------------------------------
# 10. FIVE-SENTENCE MODEL JUSTIFICATION
# --------------------------------------------------

justification = [
    (
        f"We selected {selected_name} because it achieved a test "
        f"F1 of {selected_results['f1']:.2f} for lapsed customers."
    ),
    (
        f"Its recall of {selected_results['recall']:.2f} indicates "
        "how effectively it identifies genuinely lapsed customers."
    ),
    (
        f"Its precision of {selected_results['precision']:.2f} "
        "indicates how often its lapse predictions are correct."
    ),
    (
        "Missing a genuinely lapsed customer can mean losing future "
        "orders and valuable customer lifetime value."
    ),
    (
        "Wrongly targeting a loyal customer wastes promotion budget "
        "and may encourage unnecessary discount expectations, so "
        "retention campaigns must balance both costs."
    )
]

print("\nMODEL SELECTION JUSTIFICATION")
print("=" * 55)

for sentence in justification:
    print(sentence)


# --------------------------------------------------
# 11. SAVE EVALUATION RESULTS
# --------------------------------------------------

Path("outputs").mkdir(exist_ok=True)

evaluation = pd.DataFrame([
    {
        "model": "Logistic Regression",
        "best_cv_f1": logistic_grid.best_score_,
        **logistic_results
    },
    {
        "model": "Random Forest",
        "best_cv_f1": forest_grid.best_score_,
        **forest_results
    }
])

evaluation.to_csv(
    "outputs/model_evaluation.csv",
    index=False
)

print("\nEVALUATION SAVED")
print("Output: outputs/model_evaluation.csv")
print("\nTASK 6 MODEL TRAINING AND EVALUATION COMPLETED")
