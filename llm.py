import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def llm_only_answer(question):
    response=client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role":"user","content":question}],
        temperature=0
    )
    return response.choices[0].message.content

if __name__=="__main__":
    question = "What must a Data Fiduciary do to obtain verifiable consent for a child's data?"
    print("ANSWER:", llm_only_answer(question))
