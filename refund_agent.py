
import json
import random
import re
import time

from pathlib import Path

import mock_llm

from refund_tools import TOOLS_SPEC, TOOL_MAP


# ==================================================
# 1. AGENT SYSTEM PROMPT
# ==================================================

AGENT_SYSTEM = """
AGENT

You are an internal customer support refund agent.

Use the supplied tools to investigate refund requests.

Follow these rules:

1. Always look up the order before considering a refund.
2. Search the supplied policy documents.
3. Only completed orders can be refunded.
4. Orders delivered in more than 45 minutes qualify
   for a full refund.
5. Orders delivered in 45 minutes or less do not
   qualify for a late-delivery refund.
6. Refunds above INR 1000 must be escalated to a
   human support lead.
7. Never claim that a refund was issued unless
   the issue_refund tool confirms it.
8. Never invent order information.

Use the available tools to complete each request.
"""


# ==================================================
# 2. SIMULATED CRASH EXCEPTION
# ==================================================

class SimulatedCrash(Exception):
    pass


# ==================================================
# 3. CHECKPOINT PATH
# ==================================================

def checkpoint_path(run_id):

    if not run_id or Path(run_id).name != run_id:
        raise ValueError("Invalid run ID")

    return Path("state") / f"{run_id}.json"


# ==================================================
# 4. SAVE CHECKPOINT
# ==================================================

def save_checkpoint(state):

    path = checkpoint_path(state["run_id"])

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save mock counters for deterministic recovery
    mock_state = getattr(mock_llm, "_state", None)

    if isinstance(mock_state, dict):
        state["mock_state"] = dict(mock_state)

    temporary_path = path.with_suffix(".json.tmp")

    temporary_path.write_text(
        json.dumps(
            state,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    # Replace checkpoint atomically
    temporary_path.replace(path)


# ==================================================
# 5. RESTORE MOCK STATE
# ==================================================

def restore_mock_state(state):

    saved_state = state.get("mock_state")

    current_state = getattr(mock_llm, "_state", None)

    if isinstance(saved_state, dict) and isinstance(
        current_state, dict
    ):
        current_state.clear()
        current_state.update(saved_state)


# ==================================================
# 6. BUILD RESULT
# ==================================================

def build_result(state):

    return {
        "run_id": state["run_id"],
        "scenario_id": state["scenario_id"],
        "status": state["status"],
        "final_answer": state["final_answer"],
        "refund_inr": round(state["refund_inr"], 2),
        "steps": state["steps"],
        "tools_executed": state["tools_executed"],
        "step_number": state["step_number"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "checkpoint": str(
            checkpoint_path(state["run_id"])
        )
    }


# ==================================================
# 7. REACT EXECUTION LOOP
# ==================================================

def execute_agent(
    state,
    crash_after_tool_executions=None
):

    max_iterations = state["max_iterations"]

    while state["llm_calls"] < max_iterations:

        # ------------------------------------------
        # TIME LIMIT CHECK
        # ------------------------------------------

        elapsed = time.time() - state["started_at"]

        if elapsed >= state["time_limit_seconds"]:

            state["status"] = "FALLBACK_TIMEOUT"

            state["final_answer"] = (
                "FALLBACK_TIMEOUT: Time limit reached. "
                "Handing the request to a human support agent."
            )

            save_checkpoint(state)

            return build_result(state)

        # ------------------------------------------
        # CALL MOCK MODEL
        # ------------------------------------------

        response = mock_llm.chat(
            state["messages"],
            tools=TOOLS_SPEC
        )

        state["llm_calls"] += 1

        call_tokens = response.usage.total_tokens

        state["tokens_used"] += call_tokens

        assistant_message = response.choices[0].message

        tool_calls = assistant_message.tool_calls

        # ------------------------------------------
        # TOKEN LIMIT CHECK
        # ------------------------------------------

        # Stop immediately if the budget is exceeded.
        # Do not execute the tool returned by this call.

        if state["tokens_used"] > state["token_limit"]:

            state["status"] = "FALLBACK_TOKEN_LIMIT"

            state["final_answer"] = (
                "FALLBACK_TOKEN_LIMIT: Token budget exceeded. "
                "Handing the request to a human support agent."
            )

            save_checkpoint(state)

            return build_result(state)

        # ------------------------------------------
        # FINAL ANSWER
        # ------------------------------------------

        if not tool_calls:

            state["final_answer"] = (
                assistant_message.content or ""
            )

            match = re.match(
                r"^\s*([A-Z_]+)\s*:",
                state["final_answer"]
            )

            state["status"] = (
                match.group(1)
                if match
                else "UNKNOWN"
            )

            save_checkpoint(state)

            return build_result(state)

        # ------------------------------------------
        # APPEND ASSISTANT TOOL CALL MESSAGE
        # ------------------------------------------

        assistant_tool_calls = []

        for tool_call in tool_calls:

            arguments = tool_call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            assistant_tool_calls.append({
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": arguments
            })

        state["messages"].append({
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_tool_calls
        })

        # ------------------------------------------
        # EXECUTE TOOLS
        # ------------------------------------------

        for tool_call in tool_calls:

            tool_name = tool_call.name

            arguments = tool_call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            if not isinstance(arguments, dict):
                raise TypeError(
                    f"Invalid arguments for {tool_name}"
                )

            if tool_name not in TOOL_MAP:
                raise ValueError(
                    f"Unknown tool requested: {tool_name}"
                )

            # Execute registered tool
            tool_function = TOOL_MAP[tool_name]

            observation = tool_function(**arguments)

            # Record completed tool execution
            step = {
                "step": state["step_number"] + 1,
                "tool_name": tool_name,
                "arguments": arguments,
                "observation": observation
            }

            state["steps"].append(step)

            state["tools_executed"].append(tool_name)

            state["step_number"] += 1

            # Record calculated refund amount
            if (
                tool_name == "calculate_refund"
                and isinstance(observation, dict)
                and "refund_inr" in observation
            ):
                state["refund_inr"] = float(
                    observation["refund_inr"]
                )

            # Append tool observation to conversation
            state["messages"].append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(
                    observation,
                    ensure_ascii=False
                )
            })

            # IMPORTANT:
            # Save checkpoint immediately after every tool result.
            save_checkpoint(state)

            print(
                f"CHECKPOINT SAVED: {state['run_id']} "
                f"| step={state['step_number']} "
                f"| tool={tool_name}"
            )

            # Simulated crash for Task 8 demonstration
            if (
                crash_after_tool_executions is not None
                and state["step_number"]
                == crash_after_tool_executions
            ):

                raise SimulatedCrash(
                    f"Simulated crash after tool "
                    f"execution {state['step_number']}"
                )

    # ----------------------------------------------
    # MAXIMUM ITERATIONS FALLBACK
    # ----------------------------------------------

    if state["llm_calls"] >= max_iterations:

        state["status"] = "FALLBACK_MAX_ITERATIONS"

        state["final_answer"] = (
            "FALLBACK_MAX_ITERATIONS: Maximum of "
            f"{max_iterations} model calls reached. "
            "Handing the request to a human support agent."
        )

        save_checkpoint(state)

    return build_result(state)


# ==================================================
# 8. START A NEW AGENT RUN
# ==================================================

def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tool_executions=None
):

    # New independent run
    mock_llm.reset()
    random.seed(42)

    if run_id is None:
        run_id = str(scenario_id)

    state = {
        "run_id": run_id,
        "scenario_id": str(scenario_id),

        "messages": [
            {
                "role": "system",
                "content": AGENT_SYSTEM
            },
            {
                "role": "user",
                "content": user_message
            }
        ],

        "tokens_used": 0,
        "step_number": 0,
        "tools_executed": [],
        "steps": [],

        "llm_calls": 0,
        "refund_inr": 0.0,

        "status": None,
        "final_answer": None,

        "max_iterations": max_iterations,
        "token_limit": token_limit,
        "time_limit_seconds": time_limit_seconds,

        "started_at": time.time()
    }

    # Save initial checkpoint
    save_checkpoint(state)

    return execute_agent(
        state,
        crash_after_tool_executions=crash_after_tool_executions
    )


# ==================================================
# 9. RESUME EXISTING AGENT RUN
# ==================================================

def resume_agent(run_id):

    path = checkpoint_path(run_id)

    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for run {run_id}"
        )

    # Load saved checkpoint
    state = json.loads(
        path.read_text(encoding="utf-8")
    )

    if state["run_id"] != run_id:
        raise ValueError("Checkpoint run ID mismatch")

    # Restore simulated model counters
    # Do NOT call mock_llm.reset() here.
    restore_mock_state(state)

    print("\nRESUMING AGENT")
    print("=" * 55)

    print("Run ID:", state["run_id"])
    print("Scenario:", state["scenario_id"])
    print("Completed steps:", state["step_number"])
    print("Tokens already used:", state["tokens_used"])
    print("Tools already executed:", state["tools_executed"])

    # Continue from saved messages
    return execute_agent(state)


# ==================================================
# 10. STANDALONE TEST
# ==================================================

if __name__ == "__main__":

    print("REFUND AGENT MODULE")
    print("Checkpoint and resume functions are ready.")

    print("\nAvailable functions:")
    print("run_agent(user_message, scenario_id, ...)")
    print("resume_agent(run_id)")
