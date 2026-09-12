from test_retrieval import benchmark
from llm import llm_only_answer
from rag import rag_answer
from agentic_ai_verify import verify_answer
import csv
import time

results=[]

for item in benchmark:
    ques=item["question"]
    try:
        llm_ans=llm_only_answer(ques)
        rag_ans,metadata,source_chunk=rag_answer(ques)
        verdict=verify_answer(ques,rag_ans,source_chunk)

        results.append({
            "id": item["id"],
            "category": item["category"],
            "question": ques,
            "ground_truth": item["ground_truth"],
            "llm_only_answer": llm_ans,
            "rag_answer": rag_ans,
            "verifier_verdict": verdict
        })
    except Exception as e:
        print(f"Failed on Q{item['id']}: {e}")
        continue
    time.sleep(2)

with open("benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"Saved {len(results)} results to benchmark_results.csv")