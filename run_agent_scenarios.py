
import json
import pandas as pd

from pathlib import Path

from refund_agent import run_agent


# ==================================================
# 1. CONFIGURATION
# ==================================================

SCENARIO_PATH = "data/raw/agent_scenarios.csv"

OUTPUT_DIR = Path("outputs")

RUNS_PATH = OUTPUT_DIR / "agent_runs.jsonl"
LEDGER_PATH = OUTPUT_DIR / "refund_ledger.jsonl"

OUTPUT_DIR.mkdir(exist_ok=True)


# ==================================================
# 2. LOAD SCENARIOS
# ==================================================

scenarios = pd.read_csv(
    SCENARIO_PATH,
    keep_default_na=False
)

assert len(scenarios) == 7


# ==================================================
# 3. RESET OUTPUTS FOR REPRODUCIBILITY
# ==================================================

# Start a fresh seven-scenario run
RUNS_PATH.write_text(
    "",
    encoding="utf-8"
)

# Reset ledger for this independent batch
LEDGER_PATH.write_text(
    "",
    encoding="utf-8"
)


# ==================================================
# 4. EXPECTED RESULTS
# ==================================================

expected_status = {
    "S1": "REFUND",
    "S2": "REFUND",
    "S3": "ESCALATE",
    "S4": "DENIED",
    "S5": "DENIED",
    "S6": "NOT_FOUND",
    "S7": "FALLBACK_MAX_ITERATIONS"
}

expected_refund = {
    "S1": 128.25,
    "S2": 135.00,
    "S3": 1999.00,
    "S4": 0.00,
    "S5": 0.00,
    "S6": 0.00,
    "S7": 0.00
}


# ==================================================
# 5. RUN ALL SEVEN SCENARIOS
# ==================================================

results = []

print("\nREACT REFUND AGENT SCENARIOS")
print("=" * 65)

for scenario in scenarios.itertuples(index=False):

    result = run_agent(
        user_message=scenario.message,
        scenario_id=scenario.scenario_id,
        max_iterations=6,
        token_limit=6000,
        time_limit_seconds=10
    )

    # Validate expected status
    expected = expected_status[scenario.scenario_id]

    assert result["status"] == expected, (
        f"{scenario.scenario_id}: expected {expected}, "
        f"got {result['status']}"
    )

    # Validate refund amount within INR 0.01
    expected_amount = expected_refund[
        scenario.scenario_id
    ]

    assert abs(
        result["refund_inr"] - expected_amount
    ) <= 0.01, (
        f"Refund mismatch for {scenario.scenario_id}"
    )

    # Special S1 checkpoint
    if scenario.scenario_id == "S1":

        assert result["llm_calls"] == 5, (
            "S1 must take exactly five model calls"
        )

    # S7 must use all six calls before fallback
    if scenario.scenario_id == "S7":

        assert result["llm_calls"] == 6

    # Write one JSON line per scenario
    with RUNS_PATH.open(
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )

    results.append(result)

    print("\nScenario:", scenario.scenario_id)
    print("Status:", result["status"])
    print("Final answer:", result["final_answer"])
    print("Refund INR:", result["refund_inr"])
    print("Model calls:", result["llm_calls"])
    print("Tokens used:", result["tokens_used"])

    print("\nTool steps:")

    for step in result["steps"]:

        print(
            step["step"],
            step["tool_name"],
            step["arguments"],
            "=>",
            step["observation"]
        )


# ==================================================
# 6. CHECK ALL SEVEN RESULTS
# ==================================================

print("\n" + "=" * 65)
print("SCENARIO SUMMARY")
print("=" * 65)

summary = pd.DataFrame([
    {
        "scenario_id": item["scenario_id"],
        "status": item["status"],
        "refund_inr": item["refund_inr"],
        "llm_calls": item["llm_calls"],
        "tokens_used": item["tokens_used"]
    }
    for item in results
])

print(summary.to_string(index=False))

assert len(results) == 7

# Verify output contains exactly seven JSON lines
with RUNS_PATH.open(
    "r",
    encoding="utf-8"
) as file:

    saved_runs = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

assert len(saved_runs) == 7

required_fields = {
    "scenario_id",
    "status",
    "final_answer",
    "steps",
    "llm_calls",
    "tokens_used"
}

for item in saved_runs:
    assert required_fields.issubset(item.keys())


# Save human-readable scenario summary
summary.to_csv(
    OUTPUT_DIR / "agent_scenario_summary.csv",
    index=False
)

print("\nALL SEVEN SCENARIOS PASSED")
print("Agent runs:", RUNS_PATH)
print("Scenario summary: outputs/agent_scenario_summary.csv")
