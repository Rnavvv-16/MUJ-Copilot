
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_data():
    df1 = pd.read_csv("data/academic_data.csv")
    df2 = pd.read_csv("data/student_wellbeing.csv")

    texts = []

    for _,row in df1.iterrows():
        texts.append(str(row.to_dict()))

    for _,row in df2.iterrows():
        texts.append(str(row.to_dict()))

    return texts


def build_index(texts):

    embeddings = model.encode(texts)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))

    return index, texts


def retrieve(query, index, texts):

    q_embedding = model.encode([query])

    D,I = index.search(np.array(q_embedding),3)

    results = [texts[i] for i in I[0]]

    return results
