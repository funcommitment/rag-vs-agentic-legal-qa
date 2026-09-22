from rag import rag_answer
from agentic_ai_verify import verify_answer
from embeding import collection, embedding_model

def broader_retrieve(question, n_results=5):
    q_embedding = embedding_model.encode(question)
    result = collection.query(
        query_embeddings=[q_embedding],
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    return result["documents"][0]

def agentic_answer(question, max_retries=1):
    answer, metadata, source = rag_answer(question)
    verdict = verify_answer(question, answer, source)
    
    attempts = 0
    while "no" in verdict.strip().lower()[:15] and attempts < max_retries:
        print(f"Verifier flagged issue, retrying with broader retrieval (attempt {attempts+1})...")
        wider_chunks = broader_retrieve(question, n_results=5)
        answer, metadata, source = rag_answer(question) 
        verdict = verify_answer(question, answer, "\n\n".join(wider_chunks))
        attempts += 1
    
    return answer, verdict, attempts

if __name__ == "__main__":
    question = "What is the penalty for breach in observance of additional obligations in relation to children under Section 9?"
    answer, verdict, attempts = agentic_answer(question)
    print("FINAL ANSWER:", answer)
    print("VERDICT:", verdict)
    print("RETRIES USED:", attempts)
