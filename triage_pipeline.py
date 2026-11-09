
import pandas as pd

from pathlib import Path
from mock_llm import est_tokens

# 1. Configuration
MAX_INPUT_TOKENS = 400

# 2. Load all support tickets
tickets = pd.read_csv(
    "data/raw/tickets.csv",
    keep_default_na=False
)

# 3. Estimate token count for every ticket
tickets["estimated_tokens"] = tickets["text"].apply(
    est_tokens
)

# 4. Apply input guardrails BEFORE any model call
def check_ticket(row):

    text = row["text"]

    # Empty or whitespace-only ticket
    if not text.strip():
        return "skipped_empty"

    # Ticket exceeds input token limit
    if row["estimated_tokens"] > MAX_INPUT_TOKENS:
        return "flagged_long"

    # Ticket is eligible for processing
    return "eligible"


tickets["status"] = tickets.apply(
    check_ticket,
    axis=1
)

# 5. Calculate ticket status counts
status_counts = tickets["status"].value_counts()

empty_count = int(
    (tickets["status"] == "skipped_empty").sum()
)

long_count = int(
    (tickets["status"] == "flagged_long").sum()
)

eligible_count = int(
    (tickets["status"] == "eligible").sum()
)

# 6. Validate assignment reference values
assert len(tickets) == 164
assert empty_count == 4
assert long_count == 4
assert eligible_count == 156

# All long tickets must exceed 400 tokens
long_tickets = tickets[
    tickets["status"] == "flagged_long"
]

assert (long_tickets["estimated_tokens"] > 400).all()

# The four planted long tickets should estimate to 734 tokens
assert (long_tickets["estimated_tokens"] == 734).all()

# Empty tickets must have zero estimated tokens
empty_tickets = tickets[
    tickets["status"] == "skipped_empty"
]

assert (empty_tickets["estimated_tokens"] == 0).all()

# 7. Display results
print("\nSUPPORT TICKET INPUT GUARDRAILS")
print("=" * 55)

print("Total tickets:", len(tickets))
print("Maximum allowed input tokens:", MAX_INPUT_TOKENS)

print("\nSTATUS COUNTS")
print("-" * 40)

print("Empty tickets:", empty_count)
print("Long tickets:", long_count)
print("Eligible tickets:", eligible_count)

print("\nLONG TICKET TOKEN ESTIMATES")
print(long_tickets["estimated_tokens"].tolist())

print("\nGUARDRAIL STATUS DISTRIBUTION")
print(status_counts)

# 8. Save a diagnostic report
Path("outputs").mkdir(exist_ok=True)

tickets[
    ["ticket_id", "estimated_tokens", "status"]
].to_csv(
    "outputs/ticket_guardrails.csv",
    index=False
)

print("\nALL TASK 1 CHECKPOINTS PASSED")
print("Output: outputs/ticket_guardrails.csv")
