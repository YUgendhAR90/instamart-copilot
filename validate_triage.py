
import json
import re
import random
import pandas as pd

from pathlib import Path

import mock_llm

from triage_prompts import ZERO_SHOT_SYSTEM
from triage_retry import call_with_retry

# 1. Configuration
ALLOWED_CATEGORIES = {
    "billing",
    "delivery",
    "technical",
    "fraud"
}

MAX_INPUT_TOKENS = 400

# Exact personal information patterns
PII_PATTERNS = [
    r"[\w.+-]+@[\w-]+\.[\w.-]+",
    r"\b[6-9]\d{9}\b",
    r"ORD-\d{6}"
]

# 2. Load ticket data
guardrails = pd.read_csv(
    "outputs/ticket_guardrails.csv",
    keep_default_na=False
)

redacted = pd.read_csv(
    "outputs/redacted_tickets.csv",
    keep_default_na=False
)

raw = pd.read_csv(
    "data/raw/tickets.csv",
    keep_default_na=False
)

# 3. Select eligible tickets only
eligible = guardrails[
    guardrails["status"] == "eligible"
][["ticket_id", "estimated_tokens"]].merge(
    redacted,
    on="ticket_id",
    how="inner"
).merge(
    raw[["ticket_id", "customer_id", "true_category"]],
    on="ticket_id",
    how="inner"
)

assert len(eligible) == 156

# 4. Validate privacy and input guardrails
def check_safe_text(text):

    if not str(text).strip():
        return False

    for pattern in PII_PATTERNS:
        if re.search(pattern, str(text)):
            return False

    return True


assert eligible["redacted_text"].apply(check_safe_text).all()

assert (
    eligible["estimated_tokens"] <= MAX_INPUT_TOKENS
).all()

# 5. Validate model response
def validate_response(content):

    # First attempt to parse JSON
    try:
        data = json.loads(content)

        if (
            isinstance(data, dict)
            and data.get("category") in ALLOWED_CATEGORIES
        ):
            return {
                "category": data["category"],
                "reason": data.get("reason", ""),
                "status": "classified_json"
            }

        # Valid JSON but invalid category:
        # Never use keyword fallback.
        return {
            "category": "",
            "reason": "",
            "status": "flagged_invalid_output"
        }

    except (json.JSONDecodeError, TypeError):
        pass

    # 6. Keyword fallback only for non-JSON responses
    words = re.findall(r"[a-z]+", content.lower())

    matches = [
        word for word in words
        if word in ALLOWED_CATEGORIES
    ]

    if len(matches) == 1:
        return {
            "category": matches[0],
            "reason": "Category extracted from non-JSON response.",
            "status": "classified_keyword"
        }

    # 7. Otherwise route for human review
    return {
        "category": "",
        "reason": "",
        "status": "flagged_invalid_output"
    }


# 8. Run the zero-shot classification pipeline
random.seed(42)
mock_llm.reset()

results = []

total_tokens = 0
total_retries = 0

for row in eligible.itertuples(index=False):

    # Only redacted text enters the model
    messages = [
        {
            "role": "system",
            "content": ZERO_SHOT_SYSTEM
        },
        {
            "role": "user",
            "content": row.redacted_text
        }
    ]

    response, retries = call_with_retry(
        row.ticket_id,
        messages
    )

    content = response.choices[0].message.content

    # Validate the response
    validation = validate_response(content)

    tokens_used = response.usage.total_tokens

    total_tokens += tokens_used
    total_retries += retries

    route = (
        "human_review"
        if validation["status"] == "flagged_invalid_output"
        else "pending"
    )

    results.append({
        "ticket_id": row.ticket_id,
        "customer_id": row.customer_id,
        "true_category": row.true_category,
        "category": validation["category"],
        "reason": validation["reason"],
        "status": validation["status"],
        "route": route,
        "tokens_used": tokens_used,
        "retries": retries
    })


# 9. Convert results into DataFrame
results_df = pd.DataFrame(results)

# 10. Calculate status counts
status_counts = (
    results_df["status"]
    .value_counts()
    .to_dict()
)

json_count = status_counts.get("classified_json", 0)
keyword_count = status_counts.get("classified_keyword", 0)
invalid_count = status_counts.get("flagged_invalid_output", 0)

human_count = (
    results_df["route"] == "human_review"
).sum()

# 11. Verify reference checkpoints
print("\nANSWER VALIDATION RESULTS")
print("=" * 55)

print("Eligible tickets:", len(results_df))
print("Classified JSON:", json_count)
print("Classified keyword:", keyword_count)
print("Flagged invalid output:", invalid_count)
print("Routed to human review:", human_count)

print("\nTotal response tokens:", total_tokens)
print("Total retries:", total_retries)

assert json_count == 135
assert keyword_count == 12
assert invalid_count == 9
assert human_count == 9

assert len(results_df) == 156
assert total_retries == 19

# 12. Verify all accepted categories
classified = results_df[
    results_df["status"].isin([
        "classified_json",
        "classified_keyword"
    ])
]

assert classified["category"].isin(
    ALLOWED_CATEGORIES
).all()

assert len(classified) == 147

# Invalid outputs must have human-review routing
invalid = results_df[
    results_df["status"] == "flagged_invalid_output"
]

assert (invalid["route"] == "human_review").all()

# 13. Save validation results
Path("outputs").mkdir(exist_ok=True)

results_df.to_csv(
    "outputs/triage_validation.csv",
    index=False
)

print("\nALL TASK 5 CHECKPOINTS PASSED")
print("Output: outputs/triage_validation.csv")
