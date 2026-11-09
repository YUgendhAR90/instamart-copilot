
from policy_retrieval import PolicyRetriever
import mock_llm


# ==========================================
# 1. INITIALIZE POLICY RETRIEVER
# ==========================================

retriever = PolicyRetriever(
    "outputs/policy_chunks.jsonl"
)


# ==========================================
# 2. POLICY ASSISTANT SYSTEM PROMPT
# ==========================================

POLICY_ASSISTANT_SYSTEM = """
POLICY ASSISTANT

You are an internal customer support policy assistant.

Answer questions using ONLY the policy context supplied
in the user's message.

Follow these rules:

1. Use only facts explicitly stated in the supplied context.
2. Do not invent policy rules or add external knowledge.
3. If the answer is not available in the supplied context,
   respond exactly:

   I could not find that information in the policy documents.

4. Give a clear and concise answer.
5. Do not guess when information is missing.
"""


# ==========================================
# 3. ANSWER A POLICY QUESTION USING RAG
# ==========================================

def answer(question):

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty")

    # Retrieve the three most relevant policy chunks
    retrieved = retriever.retrieve(
        question,
        k=3
    )

    # Build context with source filename
    context_parts = []

    for chunk in retrieved:

        context_parts.append(
            f"[{chunk['source']}]\n{chunk['text']}"
        )

    context = "\n".join(context_parts)

    # Build user message in required format
    user_message = (
        f"CONTEXT:\n{context}\n"
        f"QUESTION:\n{question}"
    )

    # Create the two required messages
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

    # Call supplied offline mock model
    response = mock_llm.chat(messages)

    # Extract answer
    answer_text = response.choices[0].message.content

    # Return answer, sources and token usage
    return {
        "question": question,
        "answer": answer_text,
        "sources": [
            {
                "source": chunk["source"],
                "score": chunk["score"]
            }
            for chunk in retrieved
        ],
        "total_tokens": response.usage.total_tokens
    }


# ==========================================
# 4. STANDALONE TEST
# ==========================================

if __name__ == "__main__":

    question = (
        "What delivery time makes a completed order "
        "eligible for a full refund?"
    )

    result = answer(question)

    print("\nPOLICY RAG ASSISTANT")
    print("=" * 60)

    print("\nQUESTION:")
    print(result["question"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES USED:")

    for source in result["sources"]:

        print(
            source["source"],
            "| Similarity:",
            round(source["score"], 4)
        )

    print("\nTotal tokens:", result["total_tokens"])

    # Validate the answer
    assert "more than 45 minutes" in result["answer"]

    assert len(result["sources"]) == 3

    assert any(
        source["source"] == "refund_policy.txt"
        for source in result["sources"]
    )

    print("\nALL TASK 3 CHECKPOINTS PASSED")
