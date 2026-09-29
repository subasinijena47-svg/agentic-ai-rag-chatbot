"""
Benchmark and Validation Queries Test Script
Tests the 5 core queries specified in the assignment reference guide:
1. What is Agentic AI according to the eBook?
2. How do AI agents differ from traditional automation systems?
3. What are the core components of an Agentic Architecture?
4. What role does memory play in Agentic AI workflows?
5. Who won the 2022 FIFA World Cup? (Refusal / Strict Grounding test)
"""

import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).resolve().parent))

from src.graph import build_rag_graph

BENCHMARK_QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "Who won the 2022 FIFA World Cup?",  # Expected refusal
]


def run_tests():
    print("=" * 80)
    print("RUNNING BENCHMARK & GROUNDING VERIFICATION TESTS")
    print("=" * 80)

    index_name = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
    print(f"Connecting to Pinecone index: '{index_name}'...")
    graph = build_rag_graph(index_name=index_name)

    results = []

    for idx, query in enumerate(BENCHMARK_QUERIES, 1):
        print(f"\n[{idx}/5] Query: {query}")
        print("-" * 80)

        initial_state = {
            "question": query,
            "context": [],
            "answer": "",
            "score": 0.0
        }

        try:
            output = graph.invoke(initial_state)
            ans = output["answer"]
            score = output["score"]
            context = output["context"]

            print(f"Confidence Score: {score}")
            print(f"Final Answer:\n{ans}\n")
            print(f"Retrieved Context Chunks ({len(context)}):")
            for c_idx, chunk in enumerate(context, 1):
                preview = chunk.strip().replace("\n", " ")[:150] + "..."
                print(f"  Chunk {c_idx}: {preview}")

            results.append({
                "test_id": idx,
                "query": query,
                "confidence_score": score,
                "final_answer": ans,
                "num_chunks_retrieved": len(context),
            })

        except Exception as e:
            print(f"Error during query execution: {e}")

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED.")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
