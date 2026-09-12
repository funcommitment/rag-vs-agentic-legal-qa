import os
from dotenv import load_dotenv
from groq import Groq
from embeding import collection, embedding_model


load_dotenv()
client=Groq(api_key=os.getenv("GROQ_API_KEY"))

def retrieve(question, n_results=3):
    q_embedding = embedding_model.encode(question)
    result = collection.query(
        query_embeddings=[q_embedding],
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    return result["documents"][0], result["metadatas"][0]

def build_prompt(question, chunks):
    context = "\n\n".join(chunks)
    return f"""Answer the question using ONLY the context below. 
If the answer is not in the context, say "Not found in the provided context."

Context:
{context}

Question: {question}
Answer:""",context

def rag_answer(question):
    chunks, metadatas = retrieve(question)
    prompt,context = build_prompt(question, chunks)
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content, metadatas, context

if __name__ == "__main__":
    question = "What must a Data Fiduciary do to obtain verifiable consent for a child's data?"
    answer, sources,context = rag_answer(question)
    print("ANSWER:", answer)
    print("SOURCES:", sources)
    