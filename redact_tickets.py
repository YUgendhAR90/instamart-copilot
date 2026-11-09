
import re
import pandas as pd

from pathlib import Path

# 1. Load all tickets without converting blank text into NaN
tickets = pd.read_csv(
    "data/raw/tickets.csv",
    keep_default_na=False
)

# 2. Exact assignment regular expressions
PATTERNS = {
    "email": (
        r"[\w.+-]+@[\w-]+\.[\w.-]+",
        "[EMAIL]"
    ),
    "phone": (
        r"\b[6-9]\d{9}\b",
        "[PHONE]"
    ),
    "order_id": (
        r"ORD-\d{6}",
        "[ORDER_ID]"
    )
}

# 3. Redaction function
def redact_text(text):

    counts = {}

    text = str(text)

    # Apply patterns in the exact assignment order
    for name, (pattern, replacement) in PATTERNS.items():

        text, count = re.subn(
            pattern,
            replacement,
            text
        )

        counts[name] = count

    counts["total"] = sum(counts.values())

    return text, counts


# 4. Redact every ticket
redacted_texts = []
all_counts = {
    "email": 0,
    "phone": 0,
    "order_id": 0,
    "total": 0
}

for text in tickets["text"]:

    redacted, counts = redact_text(text)

    redacted_texts.append(redacted)

    for name in all_counts:
        all_counts[name] += counts[name]


# 5. Save redacted ticket text
redacted_df = pd.DataFrame({
    "ticket_id": tickets["ticket_id"],
    "redacted_text": redacted_texts
})

Path("outputs").mkdir(exist_ok=True)

redacted_df.to_csv(
    "outputs/redacted_tickets.csv",
    index=False
)


# 6. Verify that no original patterns remain
remaining = {
    "email": 0,
    "phone": 0,
    "order_id": 0
}

for text in redacted_texts:

    for name, (pattern, replacement) in PATTERNS.items():

        matches = re.findall(pattern, text)

        remaining[name] += len(matches)

# 7. Display substitution counts
print("\nPERSONAL INFORMATION REDACTION")
print("=" * 55)

print("Total tickets:", len(tickets))

print("\nSUBSTITUTION COUNTS")

print("Emails:", all_counts["email"])
print("Phone numbers:", all_counts["phone"])
print("Order IDs:", all_counts["order_id"])
print("Total substitutions:", all_counts["total"])

print("\nREMAINING PATTERN MATCHES")

for name, count in remaining.items():
    print(f"{name}: {count}")

# 8. Validate assignment reference figures
assert len(tickets) == 164

assert all_counts["email"] == 55
assert all_counts["phone"] == 91
assert all_counts["order_id"] == 132
assert all_counts["total"] == 278

# No personal information patterns should remain
assert all(value == 0 for value in remaining.values())

assert len(redacted_df) == 164

print("\nALL TASK 2 CHECKPOINTS PASSED")
print("Output: outputs/redacted_tickets.csv")
