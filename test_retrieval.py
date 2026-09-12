from embeding import collection,embedding_model
from sentence_transformers import SentenceTransformer

#result=collection.get(limit=1, include=["embeddings", "metadatas", "documents"])
#print(result)
benchmark = [
    {
        "id": 1,
        "question": "What is the purpose of the DPDP Act?",
        "ground_truth": "To provide for the processing of digital personal data in a manner that recognises both the right of individuals to protect their personal data and the need to process such personal data for lawful purposes and for matters connected therewith or incidental thereto.",
        "source": "2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf",
        "page": 1,
        "category": "common"
    },
    {
        "id": 2,
        "question": "What is a Data Fiduciary and what is a Data Principal?",
        "ground_truth": "Data Fiduciary means any person who alone or in conjunction with other persons determines the purpose and means of processing of personal data. Data Principal means the individual to whom the personal data relates; if a child, includes the parents or lawful guardian; if a person with disability, includes her lawful guardian acting on her behalf.",
        "source": "2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf",
        "page": 2,
        "category": "common"
    },
    {
        "id": 3,
        "question": "When is a person disqualified from being appointed or continuing as Chairperson or Member of the Data Protection Board?",
        "ground_truth": "If she has been adjudged insolvent; convicted of an offence involving moral turpitude; become physically or mentally incapable; acquired financial or other interest likely to prejudicially affect her functions; or so abused her position as to render her continuance prejudicial to public interest.",
        "source": "2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf",
        "page": 13,
        "category": "specialized"
    },
    {
        "id": 4,
        "question": "What is the penalty for breach of any term of a voluntary undertaking accepted by the Board under Section 32?",
        "ground_truth": "Up to the extent applicable for the breach in respect of which the proceedings under Section 28 were instituted.",
        "source": "2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf",
        "page": 21,
        "category": "obscure"
    },
    {
        "id": 5,
        "question": "What is the penalty for breach in observance of additional obligations in relation to children under Section 9?",
        "ground_truth": "May extend to two hundred crore rupees.",
        "source": "2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf",
        "page": 21,
        "category": "obscure"
    },
    {
        "id": 6,
        "question": "When were copies of the Official Gazette containing the DPDP Rules notification made available to the public?",
        "ground_truth": "3 January 2025.",
        "source": "DPDP_Rules_2025_English_only.pdf",
        "page": 1,
        "category": "obscure"
    },
    {
        "id": 7,
        "question": "What standards apply to processing personal data for provision or issue of a subsidy, benefit, service, certificate, licence or permit by the State?",
        "ground_truth": "Processing shall be done following the standards specified in the Second Schedule. The rule also defines what 'under law', 'under policy', and 'using public funds' mean in this context.",
        "source": "DPDP_Rules_2025_English_only.pdf",
        "page": 2,
        "category": "specialized"
    },
    {
        "id": 8,
        "question": "What security safeguards must a Data Fiduciary implement to prevent a personal data breach?",
        "ground_truth": "At minimum: appropriate data security measures like encryption/obfuscation/masking/virtual tokens; access control on computer resources; logs/monitoring for detecting unauthorised access; continuity measures like data backups; retaining logs for one year; contractual safeguard provisions with Data Processors; and technical/organisational measures for compliance.",
        "source": "DPDP_Rules_2025_English_only.pdf",
        "page": 3,
        "category": "specialized"
    },
    {
        "id": 9,
        "question": "How long before completion of the erasure period must a Data Fiduciary notify the Data Principal?",
        "ground_truth": "At least forty-eight hours before completion of the time period for erasure of personal data.",
        "source": "DPDP_Rules_2025_English_only.pdf",
        "page": 4,
        "category": "specialized"
    },
    {
        "id": 10,
        "question": "What must a Data Fiduciary do to obtain verifiable consent for processing a child's personal data?",
        "ground_truth": "Adopt appropriate technical and organisational measures to ensure verifiable consent of the parent is obtained, and observe due diligence to check the parent is an identifiable adult (18+), by reference to reliable identity/age details or a virtual token issued by an authorised entity.",
        "source": "DPDP_Rules_2025_English_only.pdf",
        "page": "4-5",
        "category": "specialized"
    },
    {
        "id": 11,
        "question": "What is a 'local level committee' under the DPDP Rules?",
        "ground_truth": "A local level committee constituted under Section 13 of the National Trust for the Welfare of Persons with Autism, Cerebral Palsy, Mental Retardation and Multiple Disabilities Act, 1999.",
        "source": "DPDP_Rules_2025_English_only.pdf",
        "page": 6,
        "category": "obscure"
    },
    {
        "id": 12,
        "question": "Under what condition may personal data be transferred outside the territory of India?",
        "ground_truth": "Subject to the restriction that the Data Fiduciary shall meet such requirements as the Central Government may specify, particularly regarding making personal data available to a foreign State or any person/entity under its control.",
        "source": "DPDP_Rules_2025_English_only.pdf",
        "page": 7,
        "category": "specialized"
    },
    {
        "id": 13,
        "question": "What is the data retention period for a Data Fiduciary that is an online gaming intermediary with at least fifty lakh registered users in India (excluding account/token access purposes)?",
        "ground_truth": "Three years from the date the Data Principal last approached the Data Fiduciary for the specified purpose or exercised her rights, or the commencement of the DPDP Rules 2025, whichever is latest.",
        "source": "DPDP_Rules_2025_English_only.pdf",
        "page": 12,
        "category": "obscure"
    },
    {
        "id": 14,
        "question": "What is the processing restriction for personal data collected solely to create a user account for email communication?",
        "ground_truth": "Processing is restricted to the extent necessary for creating such user account, and its use is limited to communication by email.",
        "source": "DPDP_Rules_2025_English_only.pdf",
        "page": 14,
        "category": "obscure"
    },
    {
        "id": 15,
        "question": "What is the monthly salary of the Chairperson and other Members of the Data Protection Board?",
        "ground_truth": "The Chairperson is entitled to a consolidated salary of rupees four lakh fifty thousand per month; every other Member, rupees four lakh per month — both without house or car facility.",
        "source": "DPDP_Rules_2025_English_only.pdf",
        "page": 15,
        "category": "obscure"
    }
]
for item in benchmark:
    ques_embedding = embedding_model.encode(item["question"])
    que_result = collection.query(
        query_embeddings=[ques_embedding],
        n_results=3,
        include=["documents", "metadatas"]
    )
    print("Q:", item["question"])
    print("Expected page:", item["page"], "| source:", item["source"])
    print(que_result)
    print("-" * 50)