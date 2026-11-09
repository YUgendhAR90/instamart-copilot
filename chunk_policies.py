
import json
from pathlib import Path
from collections import Counter

# Configuration
POLICY_DIR = Path("data/policies")
OUTPUT_DIR = Path("outputs")

CHUNK_SIZE = 40
OVERLAP = 10
STEP_SIZE = CHUNK_SIZE - OVERLAP

# Expected reference counts
EXPECTED_COUNTS = {
    "refund_policy.txt": 4,
    "delivery_policy.txt": 3,
    "escalation_policy.txt": 3,
    "payment_policy.txt": 3,
    "privacy_policy.txt": 2
}


def chunk_document(source_path):

    # Read policy document
    text = source_path.read_text(encoding="utf-8")

    # Split into words
    words = text.split()

    chunks = []

    # Create overlapping chunks
    for start in range(0, len(words), STEP_SIZE):

        end = min(start + CHUNK_SIZE, len(words))

        chunk_words = words[start:end]

        chunks.append({
            "source": source_path.name,
            "start_word": start,
            "end_word": end,
            "word_count": len(chunk_words),
            "text": " ".join(chunk_words)
        })

        # Stop after the first chunk reaches the document end
        if end >= len(words):
            break

    return chunks


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    policy_files = sorted(
        POLICY_DIR.glob("*.txt")
    )

    assert len(policy_files) == 5, (
        f"Expected 5 policy documents, found {len(policy_files)}"
    )

    all_chunks = []

    # Process all five documents
    for policy_file in policy_files:

        document_chunks = chunk_document(policy_file)

        all_chunks.extend(document_chunks)

    # Add unique chunk IDs
    for index, chunk in enumerate(all_chunks, start=1):
        chunk["chunk_id"] = index

    # Count chunks by source
    source_counts = Counter(
        chunk["source"] for chunk in all_chunks
    )

    print("\nPOLICY DOCUMENT CHUNKING")
    print("=" * 55)

    print("Documents processed:", len(policy_files))

    for source, expected_count in EXPECTED_COUNTS.items():

        actual_count = source_counts.get(source, 0)

        print(
            f"{source}: {actual_count} chunks "
            f"(expected {expected_count})"
        )

        assert actual_count == expected_count, (
            f"Chunk count mismatch for {source}"
        )

    # Verify total chunks
    assert len(all_chunks) == 15

    # Save chunks as JSON Lines
    output_path = OUTPUT_DIR / "policy_chunks.jsonl"

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        for chunk in all_chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    # Save a CSV version for easy inspection
    import pandas as pd

    pd.DataFrame(all_chunks).to_csv(
        OUTPUT_DIR / "policy_chunks.csv",
        index=False
    )

    print("\nTOTAL CHUNKS:", len(all_chunks))

    print("\nALL TASK 1 CHECKPOINTS PASSED")

    print("JSON output:", output_path)
    print("CSV output: outputs/policy_chunks.csv")


if __name__ == "__main__":
    main()
