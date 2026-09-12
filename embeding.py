from sentence_transformers import SentenceTransformer
import chromadb
from splitter import splited_chunks

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

texts = [chunk.page_content for chunk in splited_chunks]
embeddings = embedding_model.encode(texts)


client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")

collection.add(
    ids=[str(i) for i in range(len(texts))],
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=[chunk.metadata for chunk in splited_chunks]
)

print(f"Stored {len(texts)} chunks in ChromaDB")