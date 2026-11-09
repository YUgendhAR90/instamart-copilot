
import json
import sqlite3

from datetime import datetime
from pathlib import Path

from policy_retrieval import PolicyRetriever


DB_PATH = "data/instamart.db"
LEDGER_PATH = Path("outputs/refund_ledger.jsonl")
REFUND_LIMIT = 1000.00

retriever = PolicyRetriever("outputs/policy_chunks.jsonl")


# ==================================================
# TOOL 1: GET ORDER
# ==================================================

def get_order(order_id):

    with sqlite3.connect(DB_PATH) as conn:

        row = conn.execute(
            """
            SELECT
                order_id,
                status,
                order_amount,
                delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,)
        ).fetchone()

    if row is None:
        return {"error": "order not found"}

    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": float(row[2]) if row[2] is not None else None,
        "delivery_minutes": float(row[3]) if row[3] is not None else None
    }


# ==================================================
# TOOL 2: SEARCH POLICY
# ==================================================

def search_policy(query):

    results = retriever.retrieve(query, k=3)

    return {"results": results}


# ==================================================
# TOOL 3: CALCULATE REFUND
# ==================================================

def calculate_refund(order_amount, delivery_minutes):

    amount = float(order_amount)
    delivery = float(delivery_minutes)

    if amount < 0:
        return {"error": "invalid order amount"}

    if delivery < 0:
        return {"error": "invalid delivery time"}

    refund = round(amount, 2) if delivery > 45 else 0.00

    return {"refund_inr": refund}


# ==================================================
# LEDGER WRITER
# ==================================================

def append_ledger(tool_name, order_id, refund_inr):

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2)
    }

    with LEDGER_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


# ==================================================
# SHARED ELIGIBILITY CHECK
# ==================================================

def verify_refund_eligibility(order_id, refund_inr):

    order = get_order(order_id)

    if "error" in order:
        return order

    if order["status"] != "completed":
        return {"error": "only completed orders can be refunded"}

    if order["order_amount"] is None:
        return {"error": "order amount unavailable"}

    if order["delivery_minutes"] is None:
        return {"error": "delivery time unavailable"}

    calculated = calculate_refund(
        order["order_amount"],
        order["delivery_minutes"]
    )

    if "error" in calculated:
        return calculated

    expected_refund = calculated["refund_inr"]
    requested_refund = round(float(refund_inr), 2)

    if expected_refund <= 0:
        return {"error": "order is not eligible for a late-delivery refund"}

    if abs(requested_refund - expected_refund) > 0.01:
        return {"error": "requested refund does not match the eligible amount"}

    return {
        "order": order,
        "refund_inr": expected_refund
    }


# ==================================================
# TOOL 4: ISSUE REFUND
# ==================================================

def issue_refund(order_id, refund_inr):

    amount = round(float(refund_inr), 2)

    if amount <= 0:
        return {"error": "refund amount must be greater than zero"}

    if amount > REFUND_LIMIT:
        return {
            "error": "refund above INR 1000 requires human approval"
        }

    verification = verify_refund_eligibility(
        order_id, amount
    )

    if "error" in verification:
        return verification

    append_ledger(
        "issue_refund",
        order_id,
        amount
    )

    return {"refund_inr": amount}


# ==================================================
# TOOL 5: ESCALATE TO HUMAN
# ==================================================

def escalate_to_human(order_id, refund_inr):

    amount = round(float(refund_inr), 2)

    if amount <= REFUND_LIMIT:
        return {
            "error": "only refunds above INR 1000 require escalation"
        }

    verification = verify_refund_eligibility(
        order_id, amount
    )

    if "error" in verification:
        return verification

    append_ledger(
        "escalate_to_human",
        order_id,
        amount
    )

    return {"refund_inr": amount}


# ==================================================
# TOOL SPECIFICATIONS
# ==================================================

TOOLS_SPEC = [

    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Retrieve order status, amount and delivery time from SQLite.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"}
                },
                "required": ["order_id"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search policy documents and return the top three chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate the refund using order amount and delivery time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number"},
                    "delivery_minutes": {"type": "number"}
                },
                "required": ["order_amount", "delivery_minutes"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issue an eligible refund up to INR 1000.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate an eligible refund above INR 1000.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    }

]


# ==================================================
# TOOL MAP
# ==================================================

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


# ==================================================
# VALIDATION
# ==================================================

if __name__ == "__main__":

    print("\nREFUND TOOL VALIDATION")
    print("=" * 55)

    assert len(TOOLS_SPEC) == 5
    assert len(TOOL_MAP) == 5

    for tool in TOOLS_SPEC:
        name = tool["function"]["name"]
        print(name)
        assert name in TOOL_MAP

    existing = get_order("ORD-100001")

    assert existing["status"] == "completed"
    print("\nORDER LOOKUP: PASSED")

    missing = get_order("ORD-999999")
    assert missing == {"error": "order not found"}
    print("MISSING ORDER: PASSED")

    policy = search_policy("refund policy late delivery")
    assert len(policy["results"]) == 3
    print("POLICY SEARCH: PASSED")

    full = calculate_refund(128.25, 65)
    none = calculate_refund(135.00, 45)

    assert full["refund_inr"] == 128.25
    assert none["refund_inr"] == 0.00
    print("REFUND CALCULATION: PASSED")

    # Cancelled order must not be refunded
    cancelled = issue_refund("ORD-100025", 45.00)

    assert "error" in cancelled
    print("CANCELLED ORDER PROTECTION: PASSED")

    # High-value automatic issuance must be rejected
    high_value = issue_refund("ORD-100145", 1999.00)

    assert "error" in high_value
    print("HIGH-VALUE REFUND PROTECTION: PASSED")

    # Small-value escalation must be rejected
    small_escalation = escalate_to_human(
        "ORD-100048", 128.25
    )

    assert "error" in small_escalation
    print("ESCALATION VALIDATION: PASSED")

    print("\nALL TOOL CHECKPOINTS PASSED")
