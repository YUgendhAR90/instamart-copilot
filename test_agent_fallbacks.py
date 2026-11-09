
import json
import pandas as pd

from pathlib import Path
from refund_agent import run_agent

# 1. Load assignment scenarios
scenarios = pd.read_csv(
    "data/raw/agent_scenarios.csv",
    keep_default_na=False
)

# Find S1 and S7
s1 = scenarios[
    scenarios["scenario_id"] == "S1"
].iloc[0]

s7 = scenarios[
    scenarios["scenario_id"] == "S7"
].iloc[0]

assert not s1.empty
assert not s7.empty


# --------------------------------------------------
# TEST A: MAXIMUM ITERATIONS
# --------------------------------------------------

print("\nTEST A - MAXIMUM ITERATIONS")
print("=" * 55)

max_result = run_agent(
    user_message=s7["message"],
    scenario_id="S7",
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10
)

print("Status:", max_result["status"])
print("Model calls:", max_result["llm_calls"])
print("Tokens used:", max_result["tokens_used"])
print("Completed steps:", len(max_result["steps"]))

assert max_result["status"] == "FALLBACK_MAX_ITERATIONS"
assert max_result["llm_calls"] == 6
assert max_result["tokens_used"] < 6000
assert "human" in max_result["final_answer"].lower()


# --------------------------------------------------
# TEST B: TOKEN LIMIT
# --------------------------------------------------

print("\nTEST B - TOKEN LIMIT")
print("=" * 55)

token_result = run_agent(
    user_message=s1["message"],
    scenario_id="S1",
    max_iterations=6,
    token_limit=200,
    time_limit_seconds=10
)

print("Status:", token_result["status"])
print("Model calls:", token_result["llm_calls"])
print("Tokens used:", token_result["tokens_used"])
print("Completed steps:", len(token_result["steps"]))

assert token_result["status"] == "FALLBACK_TOKEN_LIMIT"
assert token_result["llm_calls"] >= 1
assert token_result["tokens_used"] > 200

# Tool returned by the over-budget call must not execute
# First tool execution is preserved
assert len(token_result["steps"]) == 1

# The completed tool was the initial order lookup
assert token_result["steps"][0]["tool_name"] == "get_order"

# The token limit stopped the next tool execution
assert token_result["tokens_used"] > 200

# --------------------------------------------------
# TEST C: TIMEOUT
# --------------------------------------------------

print("\nTEST C - TIMEOUT")
print("=" * 55)

timeout_result = run_agent(
    user_message=s1["message"],
    scenario_id="S1",
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=0
)

print("Status:", timeout_result["status"])
print("Model calls:", timeout_result["llm_calls"])
print("Tokens used:", timeout_result["tokens_used"])
print("Completed steps:", len(timeout_result["steps"]))

assert timeout_result["status"] == "FALLBACK_TIMEOUT"
assert timeout_result["llm_calls"] == 0
assert timeout_result["tokens_used"] == 0
assert len(timeout_result["steps"]) == 0

assert "human" in timeout_result["final_answer"].lower()


# --------------------------------------------------
# 4. SAVE FALLBACK DEMONSTRATIONS
# --------------------------------------------------

Path("outputs").mkdir(exist_ok=True)

demo_results = [
    {
        "demo": "S7 default maximum iterations",
        **max_result
    },
    {
        "demo": "S1 token limit 200",
        **token_result
    },
    {
        "demo": "S1 timeout zero",
        **timeout_result
    }
]

with open(
    "outputs/agent_fallback_demos.jsonl",
    "w",
    encoding="utf-8"
) as file:

    for result in demo_results:

        file.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )


# --------------------------------------------------
# 5. FINAL VALIDATION
# --------------------------------------------------

print("\n" + "=" * 55)
print("SAFETY DEMONSTRATION SUMMARY")
print("=" * 55)

for result in demo_results:

    print(
        result["demo"],
        "=>",
        result["status"],
        "| Model calls:",
        result["llm_calls"]
    )

print("\nALL TASK 7 CHECKPOINTS PASSED")
print("Output: outputs/agent_fallback_demos.jsonl")
