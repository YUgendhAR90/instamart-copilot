
import json
import pandas as pd

from pathlib import Path

from refund_agent import (
    run_agent,
    resume_agent,
    SimulatedCrash
)

# ==================================================
# 1. CONFIGURATION
# ==================================================

RUN_ID = "S1-crash-resume"

RUNS_PATH = Path("outputs/agent_runs.jsonl")
LEDGER_PATH = Path("outputs/refund_ledger.jsonl")
STATE_PATH = Path("state") / f"{RUN_ID}.json"

expected_status = {
    "S1": "REFUND",
    "S2": "REFUND",
    "S3": "ESCALATE",
    "S4": "DENIED",
    "S5": "DENIED",
    "S6": "NOT_FOUND",
    "S7": "FALLBACK_MAX_ITERATIONS"
}

expected_refunds = {
    "S1": 128.25,
    "S2": 135.00,
    "S3": 1999.00,
    "S4": 0.00,
    "S5": 0.00,
    "S6": 0.00,
    "S7": 0.00
}

# ==================================================
# 2. LOAD SCENARIOS
# ==================================================

scenarios = pd.read_csv(
    "data/raw/agent_scenarios.csv",
    keep_default_na=False
)

assert len(scenarios) == 7

s1 = scenarios[
    scenarios["scenario_id"] == "S1"
].iloc[0]

assert not s1.empty

# ==================================================
# 3. START A FRESH DEMONSTRATION
# ==================================================

Path("outputs").mkdir(exist_ok=True)
Path("state").mkdir(exist_ok=True)

RUNS_PATH.write_text("", encoding="utf-8")
LEDGER_PATH.write_text("", encoding="utf-8")

# ==================================================
# 4. START S1 AND SIMULATE A CRASH
# ==================================================

print("\nTASK 8 - CRASH AND RESUME")
print("=" * 65)

try:

    run_agent(
        user_message=s1["message"],
        scenario_id="S1",
        run_id=RUN_ID,
        max_iterations=6,
        token_limit=6000,
        time_limit_seconds=10,
        crash_after_tool_executions=2
    )

    raise AssertionError(
        "The simulated crash did not occur"
    )

except SimulatedCrash as error:

    print("\nSIMULATED CRASH OCCURRED")
    print(error)

# ==================================================
# 5. INSPECT SAVED CHECKPOINT
# ==================================================

assert STATE_PATH.exists()

checkpoint = json.loads(
    STATE_PATH.read_text(encoding="utf-8")
)

print("\nSAVED CHECKPOINT")
print("-" * 55)

print("Run ID:", checkpoint["run_id"])
print("Scenario ID:", checkpoint["scenario_id"])
print("Step number:", checkpoint["step_number"])
print("Tokens used:", checkpoint["tokens_used"])
print("LLM calls:", checkpoint["llm_calls"])

print("Tools already executed:")
print(checkpoint["tools_executed"])

print("Saved messages:", len(checkpoint["messages"]))

# Verify required checkpoint contents
assert checkpoint["run_id"] == RUN_ID
assert checkpoint["scenario_id"] == "S1"

assert checkpoint["step_number"] == 2

assert checkpoint["tools_executed"] == [
    "get_order",
    "search_policy"
]

assert checkpoint["tokens_used"] > 0
assert checkpoint["llm_calls"] == 2

assert len(checkpoint["messages"]) >= 6

# ==================================================
# 6. RESUME S1 WITHOUT REPEATING TOOLS
# ==================================================

print("\nRESUMING S1")
print("-" * 55)

s1_result = resume_agent(RUN_ID)

print("\nRESUMED S1 RESULT")
print("Status:", s1_result["status"])
print("Refund:", s1_result["refund_inr"])
print("LLM calls:", s1_result["llm_calls"])

print("\nCOMPLETE TOOL EXECUTION HISTORY")

for step in s1_result["steps"]:

    print(
        step["step"],
        step["tool_name"],
        step["arguments"],
        "=>",
        step["observation"]
    )

expected_tools = [
    "get_order",
    "search_policy",
    "calculate_refund",
    "issue_refund"
]

# The four tools must execute exactly once each
assert s1_result["tools_executed"] == expected_tools

assert len(s1_result["steps"]) == 4
assert len(set(s1_result["tools_executed"])) == 4

assert s1_result["status"] == "REFUND"
assert abs(s1_result["refund_inr"] - 128.25) <= 0.01
assert s1_result["llm_calls"] == 5

# Save recovered S1 run
all_results = [s1_result]

with RUNS_PATH.open("a", encoding="utf-8") as file:

    file.write(
        json.dumps(s1_result, ensure_ascii=False) + "\n"
    )

print("\nS1 CRASH-RESUME CHECKPOINT PASSED")

# ==================================================
# 7. RUN REMAINING SCENARIOS S2-S7
# ==================================================

for scenario in scenarios.itertuples(index=False):

    scenario_id = str(scenario.scenario_id)

    if scenario_id == "S1":
        continue

    print("\nRUNNING:", scenario_id)

    result = run_agent(
        user_message=scenario.message,
        scenario_id=scenario_id,
        run_id=f"{scenario_id}-final",
        max_iterations=6,
        token_limit=6000,
        time_limit_seconds=10
    )

    # Verify expected status
    assert result["status"] == expected_status[scenario_id]

    # Verify refund amount
    assert abs(
        result["refund_inr"] - expected_refunds[scenario_id]
    ) <= 0.01

    if scenario_id == "S7":
        assert result["llm_calls"] == 6

    all_results.append(result)

    with RUNS_PATH.open("a", encoding="utf-8") as file:

        file.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )

    print("Status:", result["status"])
    print("Refund:", result["refund_inr"])
    print("Model calls:", result["llm_calls"])


# ==================================================
# 8. VERIFY EXACT FOUR TOOLS IN S1
# ==================================================

print("\nS1 TOOL EXECUTION VERIFICATION")
print("=" * 55)

print("Tools executed:", s1_result["tools_executed"])
print("Total executions:", len(s1_result["tools_executed"]))

assert s1_result["tools_executed"] == [
    "get_order",
    "search_policy",
    "calculate_refund",
    "issue_refund"
]

assert len(s1_result["tools_executed"]) == 4

# ==================================================
# 9. VERIFY REFUND LEDGER
# ==================================================

ledger_records = []

with LEDGER_PATH.open("r", encoding="utf-8") as file:

    for line in file:

        if line.strip():
            ledger_records.append(json.loads(line))

ledger_tools = [
    record["tool_name"]
    for record in ledger_records
]

print("\nREFUND LEDGER")
print("-" * 55)

for record in ledger_records:
    print(record)

assert ledger_tools == [
    "issue_refund",
    "issue_refund",
    "escalate_to_human"
]

# ==================================================
# 10. VERIFY SEVEN AGENT RUNS
# ==================================================

saved_runs = []

with RUNS_PATH.open("r", encoding="utf-8") as file:

    for line in file:

        if line.strip():
            saved_runs.append(json.loads(line))

assert len(saved_runs) == 7

summary = pd.DataFrame([
    {
        "scenario_id": result["scenario_id"],
        "status": result["status"],
        "refund_inr": result["refund_inr"],
        "llm_calls": result["llm_calls"],
        "tokens_used": result["tokens_used"]
    }
    for result in saved_runs
])

summary.to_csv(
    "outputs/agent_scenario_summary.csv",
    index=False
)

print("\nFINAL SEVEN SCENARIOS")
print("=" * 65)
print(summary.to_string(index=False))

# ==================================================
# 11. FINAL CHECKPOINT VERIFICATION
# ==================================================

final_checkpoint = json.loads(
    STATE_PATH.read_text(encoding="utf-8")
)

assert final_checkpoint["step_number"] == 4

assert final_checkpoint["tools_executed"] == expected_tools

assert final_checkpoint["status"] == "REFUND"

assert final_checkpoint["refund_inr"] == 128.25

print("\n" + "=" * 65)
print("ALL TASK 8 CHECKPOINTS PASSED")
print("=" * 65)

print("Checkpoint:", STATE_PATH)
print("Agent runs:", RUNS_PATH)
print("Refund ledger:", LEDGER_PATH)
print("Scenario summary: outputs/agent_scenario_summary.csv")
