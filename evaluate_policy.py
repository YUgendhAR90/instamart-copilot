
import json
import pandas as pd

from pathlib import Path

import mock_llm

from policy_retrieval import PolicyRetriever
from policy_assistant import POLICY_ASSISTANT_SYSTEM


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

POLICY_DIR = Path("data/policies")
OUTPUT_DIR = Path("outputs")

CHUNK_SIZE = 40
OVERLAP = 0

EVALUATION_FILE = "data/raw/policy_eval.csv"

# --------------------------------------------------
# 2. LOAD EVALUATION QUESTIONS
# --------------------------------------------------

evaluation = pd.read_csv(
    EVALUATION_FILE,
    keep_default_na=False
)

assert len(evaluation) == 14

answerable = evaluation[
    evaluation["expected_doc"] != "NONE"
].copy()

unanswerable = evaluation[
    evaluation["expected_doc"] == "NONE"
].copy()

assert len(answerable) == 12
assert len(unanswerable) == 2


# --------------------------------------------------
# 3. CREATE ZERO-OVERLAP CHUNKS
# --------------------------------------------------

def create_zero_overlap_chunks():

    chunks = []
    chunk_id = 1

    for source_path in sorted(POLICY_DIR.glob("*.txt")):

        words = source_path.read_text(
            encoding="utf-8"
        ).split()

        for start in range(0, len(words), CHUNK_SIZE):

            end = min(start + CHUNK_SIZE, len(words))

            chunk = {
                "chunk_id": chunk_id,
                "source": source_path.name,
                "start_word": start,
                "end_word": end,
                "word_count": end - start,
                "text": " ".join(words[start:end])
            }

            chunks.append(chunk)
            chunk_id += 1

            if end >= len(words):
                break

    return chunks


# Save zero-overlap chunks
zero_chunks = create_zero_overlap_chunks()

zero_chunk_path = OUTPUT_DIR / "policy_chunks_zero_overlap.jsonl"

with zero_chunk_path.open(
    "w",
    encoding="utf-8"
) as file:

    for chunk in zero_chunks:
        file.write(
            json.dumps(chunk, ensure_ascii=False) + "\n"
        )

print("Zero-overlap chunks:", len(zero_chunks))


# --------------------------------------------------
# 4. EVALUATE ONE RETRIEVAL CONFIGURATION
# --------------------------------------------------

def evaluate_retriever(retriever, experiment_name):

    print("\n" + "=" * 65)
    print(experiment_name)
    print("=" * 65)

    # Independent evaluation batch
    mock_llm.reset()

    document_hits = 0
    fact_hits = 0
    fallback_hits = 0

    details = []

    for row in evaluation.itertuples(index=False):

        question = row.question
        expected_doc = row.expected_doc
        expected_fact = row.expected_fact

        # Retrieve top three chunks
        retrieved = retriever.retrieve(
            question,
            k=3
        )

        # Build grounded context
        context_parts = []

        for chunk in retrieved:

            context_parts.append(
                f"[{chunk['source']}]\n{chunk['text']}"
            )

        context = "\n".join(context_parts)

        user_message = (
            f"CONTEXT:\n{context}\n"
            f"QUESTION:\n{question}"
        )

        messages = [
            {
                "role": "system",
                "content": POLICY_ASSISTANT_SYSTEM
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

        response = mock_llm.chat(messages)

        answer = response.choices[0].message.content

        sources = [
            result["source"]
            for result in retrieved
        ]

        # --------------------------------------------------
        # Answerable question checks
        # --------------------------------------------------

        if expected_doc != "NONE":

            doc_found = expected_doc in sources

            # Exact case-sensitive substring check
            fact_found = expected_fact in answer

            if doc_found:
                document_hits += 1

            if fact_found:
                fact_hits += 1

            result_status = (
                "PASS" if fact_found else "FAIL"
            )

        # --------------------------------------------------
        # Unanswerable question checks
        # --------------------------------------------------

        else:

            doc_found = False
            fact_found = False

            fallback_found = answer.startswith(
                "I could not find"
            )

            if fallback_found:
                fallback_hits += 1

            result_status = (
                "PASS" if fallback_found else "FAIL"
            )

        details.append({
            "question": question,
            "expected_doc": expected_doc,
            "expected_fact": expected_fact,
            "retrieved_sources": ";".join(sources),
            "answer": answer,
            "document_found": doc_found,
            "fact_found": fact_found,
            "status": result_status
        })

        print("\nQuestion:", question)
        print("Sources:", sources)
        print("Answer:", answer)
        print("Result:", result_status)

    # --------------------------------------------------
    # Evaluation summary
    # --------------------------------------------------

    print("\nEVALUATION SUMMARY")
    print("-" * 50)

    print(
        "Answerable questions:",
        len(answerable)
    )

    print(
        "Expected documents in top 3:",
        f"{document_hits}/{len(answerable)}"
    )

    print(
        "Expected facts in answers:",
        f"{fact_hits}/{len(answerable)}"
    )

    print(
        "Unanswerable fallback answers:",
        f"{fallback_hits}/{len(unanswerable)}"
    )

    return {
        "experiment": experiment_name,
        "document_hits": document_hits,
        "fact_hits": fact_hits,
        "fallback_hits": fallback_hits,
        "details": details
    }


# --------------------------------------------------
# 5. EXPERIMENT A: 10-WORD OVERLAP
# --------------------------------------------------

normal_retriever = PolicyRetriever(
    "outputs/policy_chunks.jsonl"
)

normal_results = evaluate_retriever(
    normal_retriever,
    "NORMAL CHUNKING: 40 WORDS, 10-WORD OVERLAP"
)

assert normal_results["document_hits"] == 12
assert normal_results["fact_hits"] == 12
assert normal_results["fallback_hits"] == 2


# --------------------------------------------------
# 6. EXPERIMENT B: ZERO OVERLAP
# --------------------------------------------------

zero_retriever = PolicyRetriever(
    str(zero_chunk_path)
)

zero_results = evaluate_retriever(
    zero_retriever,
    "ZERO-OVERLAP CHUNKING: 40 WORDS, ZERO OVERLAP"
)


# --------------------------------------------------
# 7. COMPARE BOTH EXPERIMENTS
# --------------------------------------------------

comparison = pd.DataFrame([
    {
        "experiment": normal_results["experiment"],
        "document_hits": normal_results["document_hits"],
        "fact_hits": normal_results["fact_hits"],
        "fallback_hits": normal_results["fallback_hits"]
    },
    {
        "experiment": zero_results["experiment"],
        "document_hits": zero_results["document_hits"],
        "fact_hits": zero_results["fact_hits"],
        "fallback_hits": zero_results["fallback_hits"]
    }
])

print("\n" + "=" * 65)
print("OVERLAP COMPARISON")
print("=" * 65)

print(comparison.to_string(index=False))


# --------------------------------------------------
# 8. IDENTIFY ZERO-OVERLAP FAILURES
# --------------------------------------------------

print("\nZERO-OVERLAP FAILURES")
print("-" * 55)

failures = [
    item for item in zero_results["details"]
    if item["expected_doc"] != "NONE"
    and item["status"] == "FAIL"
]

for failure in failures:

    print("\nQuestion:", failure["question"])
    print("Expected fact:", failure["expected_fact"])
    print("Retrieved sources:", failure["retrieved_sources"])
    print("Generated answer:", failure["answer"])

print("\nNumber of answerable failures:", len(failures))

# Assignment requires at least one failure without overlap
assert len(failures) >= 1, (
    "Expected at least one answerable question to fail "
    "with zero overlap. Inspect retrieval and chunk boundaries."
)


# --------------------------------------------------
# 9. SAVE EVALUATION RESULTS
# --------------------------------------------------

OUTPUT_DIR.mkdir(exist_ok=True)

pd.DataFrame(normal_results["details"]).to_csv(
    "outputs/policy_eval_overlap.csv",
    index=False
)

pd.DataFrame(zero_results["details"]).to_csv(
    "outputs/policy_eval_zero_overlap.csv",
    index=False
)

comparison.to_csv(
    "outputs/policy_eval_comparison.csv",
    index=False
)

print("\nALL TASK 4 CHECKPOINTS PASSED")

print("Normal evaluation: outputs/policy_eval_overlap.csv")
print("Zero-overlap evaluation: outputs/policy_eval_zero_overlap.csv")
print("Comparison: outputs/policy_eval_comparison.csv")
