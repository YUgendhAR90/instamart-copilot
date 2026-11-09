
# ==========================================
# ZERO-SHOT CLASSIFICATION PROMPT
# ==========================================

ZERO_SHOT_SYSTEM = """
CLASSIFY customer support tickets.

You are a customer support ticket classification assistant.

Classify each ticket into exactly one of these four categories:

1. billing
2. delivery
3. technical
4. fraud

Use the meaning and keywords of the ticket to determine
the most appropriate category.

The reason must briefly explain why the category was selected.

Return ONLY a valid JSON object in this exact format:

{"category": "...", "reason": "..."}

Do not include markdown, explanations outside JSON,
or any additional text.

The category must be exactly one of:
billing, delivery, technical, fraud.
"""


# ==========================================
# FEW-SHOT CLASSIFICATION PROMPT
# ==========================================

FEW_SHOT_SYSTEM = """
CLASSIFY customer support tickets.

You are a customer support ticket classification assistant.

Classify each ticket into exactly one of these four categories:

1. billing
2. delivery
3. technical
4. fraud

Use the meaning and keywords of the ticket to determine
the most appropriate category.

The reason must briefly explain why the category was selected.

Return ONLY a valid JSON object in this exact format:

{"category": "...", "reason": "..."}

Do not include markdown, explanations outside JSON,
or any additional text.

EXAMPLES:

Example 1 - BILLING

Ticket: I was charged twice for my order and need a refund.

Expected response:
{"category": "billing", "reason": "The customer reports a duplicate charge."}


Example 2 - FRAUD

Ticket: Someone made an unauthorized transaction using my account.

Expected response:
{"category": "fraud", "reason": "The customer reports an unauthorized transaction."}


Example 3 - DELIVERY

Ticket: My order has not arrived and the rider is late.

Expected response:
{"category": "delivery", "reason": "The customer reports a delayed delivery."}


Example 4 - TECHNICAL

Ticket: The application crashes whenever I try to open it.

Expected response:
{"category": "technical", "reason": "The customer reports an application crash."}


Follow the instructions and output only the required JSON object.
"""


# ==========================================
# PROMPT VALIDATION
# ==========================================

ALLOWED_CATEGORIES = [
    "billing",
    "delivery",
    "technical",
    "fraud"
]

JSON_FORMAT = '{"category": "...", "reason": "..."}'


def validate_prompts():

    # Both prompts must contain CLASSIFY
    assert "CLASSIFY" in ZERO_SHOT_SYSTEM
    assert "CLASSIFY" in FEW_SHOT_SYSTEM

    # Both must contain the required JSON format
    assert JSON_FORMAT in ZERO_SHOT_SYSTEM
    assert JSON_FORMAT in FEW_SHOT_SYSTEM

    # Verify all allowed categories
    for category in ALLOWED_CATEGORIES:

        assert category in ZERO_SHOT_SYSTEM
        assert category in FEW_SHOT_SYSTEM

    # Zero-shot must not contain EXAMPLES:
    assert "EXAMPLES:" not in ZERO_SHOT_SYSTEM

    # Few-shot must contain EXAMPLES: as a separate line
    assert "EXAMPLES:" in FEW_SHOT_SYSTEM.splitlines()

    # Verify labelled examples
    assert "Example 1 - BILLING" in FEW_SHOT_SYSTEM
    assert "Example 2 - FRAUD" in FEW_SHOT_SYSTEM

    # Verify the fraud example
    assert '"category": "fraud"' in FEW_SHOT_SYSTEM

    print("ZERO-SHOT PROMPT: PASSED")
    print("FEW-SHOT PROMPT: PASSED")
    print("ALLOWED CATEGORIES: PASSED")
    print("JSON FORMAT: PASSED")
    print("EXAMPLE VALIDATION: PASSED")
    print("FRAUD EXAMPLE: PASSED")

    print("\nALL TASK 3 CHECKPOINTS PASSED")


if __name__ == "__main__":
    validate_prompts()
