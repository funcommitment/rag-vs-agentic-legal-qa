from langchain_core.documents import Document
import pdfplumber
import pandas as pd

def load_file(file_name):
    if file_name.split('.')[-1].lower()!="pdf":
        raise ValueError("File should be in pdf format")
    
    documents=[]
    with pdfplumber.open(file_name) as pdf:
        for page_num,page in enumerate(pdf.pages,start=1):
            text=page.extract_text()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source":file_name,
                            "page":page_num,
                            "text":"text"

                        }
                    )
                )
            
    return documents

output1=load_file("DPDP_Rules_2025_English_only.pdf")
output2=load_file("2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf")
final_load=output2+output1
