import os
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from transformers import pipeline


def load_documents():

    folder = "data"
    documents = []

    for file in os.listdir(folder):

        if file.endswith(".txt"):

            file_path = os.path.join(folder, file)

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            documents.append({
                "filename": file,
                "text": text
            })

    return documents


def clean_text(text):

    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # remove page numbers
    text = re.sub(r'Page \d+', '', text)

    return text.strip()


def prepare_chunks(documents, chunk_size=800, overlap=150):

    chunks = []

    for doc in documents:

        text = clean_text(doc["text"])

        start = 0

        while start < len(text):

            chunk_text = text[start:start+chunk_size]

            chunks.append({
                "filename": doc["filename"],
                "text": chunk_text
            })

            start += chunk_size - overlap

    return chunks




def create_vector_store(chunks):

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embed_model.encode(texts)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    llm = pipeline(
        "text-generation",
        model="google/flan-t5-small"
    )

    return index, llm, chunks

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
   
def search(query, index, chunk_metadata, k=5):

    query_vector = embed_model.encode([query])

    query_vector = np.array(query_vector).astype("float32")

    distances, indices = index.search(query_vector, k)

    results = []

    for i, dist in zip(indices[0], distances[0]):
        if dist < 1.5:   # filter weak matches
            results.append(chunk_metadata[i])
    if len(results) == 0:
        for i in indices[0][:2]:
            results.append(chunk_metadata[i])

    return results




def generate_answer(query, retrieved_chunks, model):

    context = "\n\n".join([chunk["text"] [:400] for chunk in retrieved_chunks])

    prompt = f"""
Use legal context below to answer the question briefly.

Context:
{context}

Question:
{query}

Provide a short legal explanation in 3 to 4 sentences. 
"""

    result = model(prompt, max_length=180, do_sample=False)

    answer = result[0]["generated_text"].strip()


    return answer







