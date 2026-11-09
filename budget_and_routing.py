
import json
import re
import random

import pandas as pd

from pathlib import Path

import mock_llm

from triage_prompts import (
    ZERO_SHOT_SYSTEM,
    FEW_SHOT_SYSTEM
)

from triage_retry import call_with_retry


# ==================================================
# 1. CONFIGURATION
# ==================================================

ZERO_SHOT_BUDGET = 6000
FEW_SHOT_BUDGET = 100000

ALLOWED_CATEGORIES = {
    "billing",
    "delivery",
    "technical",
    "fraud"
}

OUTPUT_COLUMNS = [
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used"
]


# ==================================================
# 2. LOAD PROJECT DATA
# ==================================================

raw = pd.read_csv(
    "data/raw/tickets.csv",
    keep_default_na=False
)

guardrails = pd.read_csv(
    "outputs/ticket_guardrails.csv",
    keep_default_na=False
)

redacted = pd.read_csv(
    "outputs/redacted_tickets.csv",
    keep_default_na=False
)

segments = pd.read_csv(
    "outputs/customer_segments.csv",
    keep_default_na=False
)

# Create lookup dictionaries
guard_map = (
    guardrails.set_index("ticket_id").to_dict(orient="index")
)

redacted_map = dict(
    zip(redacted["ticket_id"], redacted["redacted_text"])
)

segment_name_map = dict(
    zip(segments["customer_id"], segments["segment_name"])
)

# Safely interpret Premium flags
premium_map = {
    row.customer_id: str(row.is_premium).strip().lower()
    in {"true", "1", "yes"}
    for row in segments[
        ["customer_id", "is_premium"]
    ].itertuples(index=False)
}

assert len(raw) == 164
assert len(guardrails) == 164
assert len(redacted) == 164
assert len(segments) == 1175


# ==================================================
# 3. PRIVACY CHECK
# ==================================================

PII_PATTERNS = [
    r"[\w.+-]+@[\w-]+\.[\w.-]+",
    r"\b[6-9]\d{9}\b",
    r"ORD-\d{6}"
]


def is_safe(text):

    if not str(text).strip():
        return False

    return not any(
        re.search(pattern, str(text))
        for pattern in PII_PATTERNS
    )


# ==================================================
# 4. RESPONSE VALIDATION
# ==================================================

def validate_response(content):

    # First, try JSON parsing
    try:
        data = json.loads(content)

        if (
            isinstance(data, dict)
            and isinstance(data.get("category"), str)
            and data["category"] in ALLOWED_CATEGORIES
        ):
            return (
                data["category"],
                "classified_json"
            )

        # Valid JSON but invalid category
        return "", "flagged_invalid_output"

    except (json.JSONDecodeError, TypeError):
        pass

    # Extract words from non-JSON responses
    words = re.findall(
        r"[a-z]+",
        content.lower()
    )

    matches = [
        word for word in words
        if word in ALLOWED_CATEGORIES
    ]

    if len(matches) == 1:
        return matches[0], "classified_keyword"

    return "", "flagged_invalid_output"


# ==================================================
# 5. ROUTING FUNCTION
# ==================================================

def determine_route(status, category, is_premium):

    # Human review always takes precedence
    if status in {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted"
    }:
        return "NORMAL", "human_review"

    # Fraud takes precedence over Premium
    if category == "fraud":
        return "HIGH", "security_team"

    # Premium customer
    if is_premium:
        return "HIGH", "premium_desk"

    return "NORMAL", "support_queue"


# ==================================================
# 6. COMPLETE PIPELINE WITH TOKEN BUDGET
# ==================================================

def run_pipeline(prompt_name, system_prompt, token_budget):

    print("\n" + "=" * 65)
    print(prompt_name.upper())
    print("=" * 65)

    # Reset mock for independent full run
    random.seed(42)
    mock_llm.reset()

    running_tokens = 0
    retries_total = 0
    successful_calls = 0
    exhausted_count = 0

    results = []

    # Preserve original ticket order
    for ticket in raw.itertuples(index=False):

        ticket_id = ticket.ticket_id
        customer_id = ticket.customer_id

        guard = guard_map[ticket_id]
        original_status = guard["status"]

        segment_name = segment_name_map.get(
            customer_id,
            ""
        )

        is_premium = premium_map.get(
            customer_id,
            False
        )

        category = ""
        tokens_used = 0

        # ------------------------------------------
        # INPUT GUARDRAILS
        # ------------------------------------------

        if original_status in {
            "skipped_empty",
            "flagged_long"
        }:

            status = original_status

        # ------------------------------------------
        # TOKEN BUDGET
        # ------------------------------------------

        elif running_tokens >= token_budget:

            status = "budget_exhausted"
            exhausted_count += 1

        # ------------------------------------------
        # MODEL CALL
        # ------------------------------------------

        else:

            user_text = redacted_map[ticket_id]

            # Privacy assertion before every model call
            assert is_safe(user_text)

            assert guard["estimated_tokens"] <= 400

            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_text
                }
            ]

            response, retries = call_with_retry(
                ticket_id,
                messages
            )

            retries_total += retries
            successful_calls += 1

            # Count actual response usage
            tokens_used = response.usage.total_tokens

            running_tokens += tokens_used

            content = response.choices[0].message.content

            category, status = validate_response(content)

        # ------------------------------------------
        # FINAL ROUTING
        # ------------------------------------------

        priority, route = determine_route(
            status,
            category,
            is_premium
        )

        results.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": tokens_used
        })

    result_df = pd.DataFrame(
        results,
        columns=OUTPUT_COLUMNS
    )

    # ----------------------------------------------
    # VALIDATION
    # ----------------------------------------------

    assert len(result_df) == 164

    assert (result_df["route"] == "human_review").sum() >= 17

    assert result_df["category"].fillna("").apply(
        lambda x: x == "" or x in ALLOWED_CATEGORIES
    ).all()

    assert result_df.loc[
        result_df["route"] == "human_review",
        "priority"
    ].eq("NORMAL").all()

    assert successful_calls + 8 + exhausted_count == 164

    return result_df, {
        "tokens_used": running_tokens,
        "successful_calls": successful_calls,
        "retries": retries_total,
        "budget_exhausted": exhausted_count
    }


# ==================================================
# 7. ZERO-SHOT BUDGET DEMONSTRATION
# ==================================================

demo, demo_stats = run_pipeline(
    "ZERO-SHOT BUDGET DEMONSTRATION",
    ZERO_SHOT_SYSTEM,
    ZERO_SHOT_BUDGET
)

demo_path = "outputs/triage_budget_demo.csv"

demo.to_csv(
    demo_path,
    index=False
)

demo_exhausted = int(
    (demo["status"] == "budget_exhausted").sum()
)

assert demo_exhausted >= 1

print("\nZERO-SHOT BUDGET RESULTS")
print("=" * 55)

print("Budget:", ZERO_SHOT_BUDGET)
print("Response tokens used:", demo_stats["tokens_used"])
print("Successful model calls:", demo_stats["successful_calls"])
print("Recovered retries:", demo_stats["retries"])
print("Budget-exhausted tickets:", demo_exhausted)

print("Output:", demo_path)


# ==================================================
# 8. FINAL FEW-SHOT RUN
# ==================================================

final, final_stats = run_pipeline(
    "FINAL FEW-SHOT TRIAGE RUN",
    FEW_SHOT_SYSTEM,
    FEW_SHOT_BUDGET
)

final_path = "outputs/triage_results.csv"

final.to_csv(
    final_path,
    index=False
)

# ----------------------------------------------
# Validate all final routing rules
# ----------------------------------------------

for row in final.itertuples(index=False):

    if row.status in {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted"
    }:

        assert row.route == "human_review"
        assert row.priority == "NORMAL"

    elif row.category == "fraud":

        assert row.route == "security_team"
        assert row.priority == "HIGH"

    elif premium_map.get(row.customer_id, False):

        assert row.route == "premium_desk"
        assert row.priority == "HIGH"

    else:

        assert row.route == "support_queue"
        assert row.priority == "NORMAL"


# Final acceptance checks
assert len(final) == 164
assert final["ticket_id"].is_unique

assert list(final.columns) == OUTPUT_COLUMNS

# Full few-shot run should fit within budget
assert final_stats["tokens_used"] < FEW_SHOT_BUDGET
assert final_stats["budget_exhausted"] == 0

# Reference status counts
final_status_counts = final["status"].value_counts().to_dict()

assert final_status_counts.get("skipped_empty", 0) == 4
assert final_status_counts.get("flagged_long", 0) == 4
assert final_status_counts.get("classified_json", 0) == 135
assert final_status_counts.get("classified_keyword", 0) == 12
assert final_status_counts.get("flagged_invalid_output", 0) == 9


# ==================================================
# 9. FINAL SUMMARY
# ==================================================

print("\nFINAL FEW-SHOT RESULTS")
print("=" * 65)

print("Total tickets:", len(final))
print("\nTicket status counts:")
print(final["status"].value_counts())

print("\nRouting counts:")
print(final["route"].value_counts())

print("\nPriority counts:")
print(final["priority"].value_counts())

print("\nResponse tokens used:", final_stats["tokens_used"])
print("Successful model calls:", final_stats["successful_calls"])
print("Recovered retries:", final_stats["retries"])

print("\nFINAL OUTPUT")
print(final_path)

print("\n" + "=" * 65)
print("ALL TASK 7 CHECKPOINTS PASSED")
print("=" * 65)
