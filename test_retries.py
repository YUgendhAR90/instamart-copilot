
import random
import pandas as pd

import mock_llm

from triage_retry import call_with_retry
from triage_prompts import ZERO_SHOT_SYSTEM

# 1. Load guardrail results
guardrails = pd.read_csv(
    "outputs/ticket_guardrails.csv"
)

# 2. Load redacted ticket text
redacted = pd.read_csv(
    "outputs/redacted_tickets.csv",
    keep_default_na=False
)

# 3. Select only eligible tickets
eligible = guardrails[
    guardrails["status"] == "eligible"
][["ticket_id"]].merge(
    redacted,
    on="ticket_id",
    how="inner",
    sort=False
)

assert len(eligible) == 156

# Counters
successful_calls = 0
total_transient_errors = 0
recovered_tickets = 0
permanent_errors = []

# Deterministic jitter for reproducibility
random.seed(42)

# Reset mock immediately before the full run
mock_llm.reset()

# 4. Process all eligible tickets
for row in eligible.itertuples(index=False):

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

    try:

        response, retries = call_with_retry(
            row.ticket_id,
            messages
        )

        successful_calls += 1
        total_transient_errors += retries

        if retries > 0:
            recovered_tickets += 1

    except mock_llm.TransientLLMError:

        permanent_errors.append(row.ticket_id)

# 5. Display results
print("\nRESILIENT CALL RESULTS")
print("=" * 55)

print("Eligible tickets:", len(eligible))
print("Successful calls:", successful_calls)
print("Transient errors:", total_transient_errors)
print("Recovered tickets:", recovered_tickets)
print("Permanent errors:", len(permanent_errors))

print("\nPermanent error ticket IDs:", permanent_errors)

# 6. Validate assignment references
assert successful_calls == 156
assert total_transient_errors == 19
assert recovered_tickets == 19
assert len(permanent_errors) == 0

print("\nALL TASK 4 CHECKPOINTS PASSED")
