
import json
import re
import random
import pandas as pd

from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix

import mock_llm

from triage_prompts import (
    ZERO_SHOT_SYSTEM,
    FEW_SHOT_SYSTEM
)

from triage_retry import call_with_retry

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

ALLOWED_CATEGORIES = {
    "billing",
    "delivery",
    "technical",
    "fraud"
}

# --------------------------------------------------
# 2. LOAD TICKETS
# --------------------------------------------------

guardrails = pd.read_csv(
    "outputs/ticket_guardrails.csv"
)

redacted = pd.read_csv(
    "outputs/redacted_tickets.csv",
    keep_default_na=False
)

raw = pd.read_csv(
    "data/raw/tickets.csv",
    keep_default_na=False
)

eligible = guardrails[
    guardrails["status"] == "eligible"
][["ticket_id", "estimated_tokens"]].merge(
    redacted,
    on="ticket_id",
    how="inner",
    sort=False
).merge(
    raw[["ticket_id", "true_category"]],
    on="ticket_id",
    how="inner",
    sort=False
)

assert len(eligible) == 156


# --------------------------------------------------
# 3. VALIDATE MODEL RESPONSES
# --------------------------------------------------

def validate_response(content):

    try:
        data = json.loads(content)

        if (
            isinstance(data, dict)
            and isinstance(data.get("category"), str)
            and data["category"] in ALLOWED_CATEGORIES
        ):
            return data["category"], "classified_json"

        # Valid JSON with an invalid category
        return "", "flagged_invalid_output"

    except (json.JSONDecodeError, TypeError):
        pass

    # Extract lowercase words from non-JSON responses
    words = re.findall(r"[a-z]+", content.lower())

    matches = [
        word for word in words
        if word in ALLOWED_CATEGORIES
    ]

    if len(matches) == 1:
        return matches[0], "classified_keyword"

    return "", "flagged_invalid_output"


# --------------------------------------------------
# 4. RUN ONE COMPLETE PROMPT EXPERIMENT
# --------------------------------------------------

def run_experiment(prompt_name, system_prompt):

    print("\n" + "=" * 60)
    print(prompt_name.upper(), "EXPERIMENT")
    print("=" * 60)

    # Reset the supplied mock for an independent run
    random.seed(42)
    mock_llm.reset()

    rows = []
    total_tokens = 0
    total_retries = 0

    for ticket in eligible.itertuples(index=False):

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": ticket.redacted_text
            }
        ]

        response, retries = call_with_retry(
            ticket.ticket_id,
            messages
        )

        content = response.choices[0].message.content

        category, status = validate_response(content)

        # Use actual mock response token usage
        tokens = response.usage.total_tokens

        total_tokens += tokens
        total_retries += retries

        rows.append({
            "ticket_id": ticket.ticket_id,
            "true_category": ticket.true_category,
            "predicted_category": category,
            "status": status,
            "tokens_used": tokens,
            "retries": retries
        })

    result = pd.DataFrame(rows)

    # Only successfully classified tickets count for accuracy
    classified = result[
        result["status"].isin([
            "classified_json",
            "classified_keyword"
        ])
    ]

    invalid_count = (
        result["status"] == "flagged_invalid_output"
    ).sum()

    json_count = (
        result["status"] == "classified_json"
    ).sum()

    keyword_count = (
        result["status"] == "classified_keyword"
    ).sum()

    accuracy = accuracy_score(
        classified["true_category"],
        classified["predicted_category"]
    )

    tokens_per_ticket = (
        total_tokens / len(classified)
    )

    print("\nCLASSIFICATION COUNTS")
    print("Total attempted:", len(result))
    print("Classified JSON:", json_count)
    print("Classified keyword:", keyword_count)
    print("Invalid output:", invalid_count)
    print("Total classified:", len(classified))

    print("\nACCURACY")
    print(
        f"Correct: "
        f"{(classified['true_category'] == classified['predicted_category']).sum()}"
        f" / {len(classified)}"
    )
    print("Accuracy:", f"{accuracy * 100:.1f}%")

    print("\nTOKEN USAGE")
    print("Total response tokens:", total_tokens)
    print("Tokens per classified ticket:", round(tokens_per_ticket, 2))

    print("Recovered retries:", total_retries)

    assert len(result) == 156
    assert len(classified) == 147
    assert json_count == 135
    assert keyword_count == 12
    assert invalid_count == 9
    assert total_retries == 19

    return result, classified, {
        "prompt": prompt_name,
        "tickets_classified": len(classified),
        "correct": int(
            (
                classified["true_category"]
                == classified["predicted_category"]
            ).sum()
        ),
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "tokens_per_ticket": tokens_per_ticket,
        "json_count": int(json_count),
        "keyword_count": int(keyword_count),
        "invalid_count": int(invalid_count)
    }


# --------------------------------------------------
# 5. ZERO-SHOT EXPERIMENT
# --------------------------------------------------

zero_result, zero_classified, zero_metrics = run_experiment(
    "Zero-shot",
    ZERO_SHOT_SYSTEM
)


# --------------------------------------------------
# 6. ZERO-SHOT CONFUSION MATRIX
# --------------------------------------------------

print("\nZERO-SHOT CONFUSION MATRIX")
print("=" * 60)

labels = [
    "billing",
    "delivery",
    "technical",
    "fraud"
]

matrix = confusion_matrix(
    zero_classified["true_category"],
    zero_classified["predicted_category"],
    labels=labels
)

matrix_df = pd.DataFrame(
    matrix,
    index=labels,
    columns=labels
)

print("\nRows = Actual categories")
print("Columns = Predicted categories\n")
print(matrix_df)

# Check the reference error pattern
errors = zero_classified[
    zero_classified["true_category"]
    != zero_classified["predicted_category"]
]

error_counts = (
    errors.groupby(
        ["true_category", "predicted_category"]
    )
    .size()
    .to_dict()
)

print("\nZERO-SHOT ERROR BREAKDOWN")
print(error_counts)

assert len(errors) == 21

assert error_counts == {
    ("fraud", "billing"): 11,
    ("technical", "billing"): 10
}

assert round(zero_metrics["accuracy"] * 100, 1) == 85.7


# --------------------------------------------------
# 7. FEW-SHOT EXPERIMENT
# --------------------------------------------------

few_result, few_classified, few_metrics = run_experiment(
    "Few-shot",
    FEW_SHOT_SYSTEM
)

assert few_metrics["accuracy"] == 1.0


# --------------------------------------------------
# 8. COMPARE BOTH EXPERIMENTS
# --------------------------------------------------

print("\nZERO-SHOT VS FEW-SHOT COMPARISON")
print("=" * 70)

comparison = pd.DataFrame([
    zero_metrics,
    few_metrics
])

print(
    comparison[
        [
            "prompt",
            "tickets_classified",
            "correct",
            "accuracy",
            "total_tokens",
            "tokens_per_ticket"
        ]
    ].to_string(
        index=False,
        formatters={
            "accuracy": lambda x: f"{x * 100:.1f}%",
            "tokens_per_ticket": lambda x: f"{x:.2f}"
        }
    )
)


# --------------------------------------------------
# 9. EXPLAIN TOKEN COST
# --------------------------------------------------

extra_tokens = (
    few_metrics["total_tokens"]
    - zero_metrics["total_tokens"]
)

print("\nADDITIONAL FEW-SHOT TOKEN COST")
print("Extra tokens:", extra_tokens)

print(
    "Extra tokens per classified ticket:",
    round(
        few_metrics["tokens_per_ticket"]
        - zero_metrics["tokens_per_ticket"],
        2
    )
)


# --------------------------------------------------
# 10. SAVE RESULTS
# --------------------------------------------------

Path("outputs").mkdir(exist_ok=True)

comparison.to_csv(
    "outputs/prompt_comparison.csv",
    index=False
)

zero_result.to_csv(
    "outputs/zero_shot_results.csv",
    index=False
)

few_result.to_csv(
    "outputs/few_shot_results.csv",
    index=False
)

matrix_df.to_csv(
    "outputs/zero_shot_confusion.csv"
)

print("\nALL TASK 6 CHECKPOINTS PASSED")
print("Outputs saved in outputs/")
