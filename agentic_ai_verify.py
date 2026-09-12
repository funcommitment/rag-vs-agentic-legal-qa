import os
from dotenv import load_dotenv
from groq import Groq
from rag import rag_answer

load_dotenv()
client=Groq(api_key=os.getenv("GROQ_API_KEY"))

def verify_answer(question,answer,source):
    prompt=f'''Here's the question {question} and here's the answer {answer}.
    In response tell yes or no if the answer properly answers the question and if it hallucinates or answers directly from the source and here is the source {source} and also the reasoning.'''
    response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
    return response.choices[0].message.content


if __name__ == "__main__":
    question = "What must a Data Fiduciary do to obtain verifiable consent for a child's data?"
    answer, metadata,context_source= rag_answer(question)
    answer_verified=verify_answer(question,answer,context_source)
    print(answer_verified)