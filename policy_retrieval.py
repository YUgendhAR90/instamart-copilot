
import json
import numpy as np

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PolicyRetriever:

    def __init__(self, chunk_path="outputs/policy_chunks.jsonl"):

        # 1. Load policy chunks
        self.chunks = []

        with open(chunk_path, "r", encoding="utf-8") as file:

            for line in file:
                self.chunks.append(json.loads(line))

        # 2. Extract chunk text
        self.texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        # 3. Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer()

        # 4. Fit vectorizer on policy chunks
        self.document_matrix = (
            self.vectorizer.fit_transform(self.texts)
        )

    # 5. Retrieve the most relevant chunks
    def retrieve(self, query, k=3):

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must not be empty")

        if k < 1:
            raise ValueError("k must be at least 1")

        # Transform query using the SAME fitted vectorizer
        query_vector = self.vectorizer.transform([query])

        # Calculate cosine similarities
        scores = cosine_similarity(
            query_vector,
            self.document_matrix
        )[0]

        # Rank chunks by similarity, highest first
        ranked_indices = np.argsort(
            -scores,
            kind="stable"
        )[:k]

        # Return required dictionary format
        results = []

        for index in ranked_indices:

            results.append({
                "source": self.chunks[index]["source"],
                "score": float(scores[index]),
                "text": self.chunks[index]["text"]
            })

        return results


# --------------------------------------------------
# TEST RETRIEVAL
# --------------------------------------------------

if __name__ == "__main__":

    retriever = PolicyRetriever()

    query = "refund policy late delivery"

    results = retriever.retrieve(
        query,
        k=3
    )

    print("\nPOLICY RETRIEVAL RESULTS")
    print("=" * 60)

    print("Query:", query)
    print("Total policy chunks:", len(retriever.chunks))

    for rank, result in enumerate(results, start=1):

        print(f"\nRank {rank}")
        print("Source:", result["source"])
        print("Similarity:", round(result["score"], 4))
        print("Text:", result["text"])

    # Assignment reference validation
    assert len(results) == 3

    assert all(
        result["source"] == "refund_policy.txt"
        for result in results
    ), "Top three results must come from refund_policy.txt"

    print("\nALL TASK 2 CHECKPOINTS PASSED")
