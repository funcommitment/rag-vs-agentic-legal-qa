from loaders import final_load
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_chunk(loaded_file):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(loaded_file)

splited_chunks=split_chunk(final_load)
print(len(splited_chunks))