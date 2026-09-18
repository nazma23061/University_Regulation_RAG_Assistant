import os
import pickle
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


data_folder = "data"

chunks = []

chunk_size = 1000


# Read all PDF files from data folder
for filename in os.listdir(data_folder):

    if filename.endswith(".pdf"):

        pdf_path = os.path.join(data_folder, filename)

        reader = PdfReader(pdf_path)

        print("Loading:", filename)

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text()

            if text:

                text = text.replace("\n", " ")

                for i in range(0, len(text), chunk_size):

                    chunk = text[i:i + chunk_size]

                    if len(chunk.strip()) > 20:

                        chunks.append({
                            "text": chunk,
                            "page": page_number,
                            "source": filename
                        })


print("Total chunks:", len(chunks))


# Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(texts)


print("Embeddings created!")
print("Number of embeddings:", len(embeddings))
print("Embedding size:", len(embeddings[0]))


# Create FAISS index
dimension = len(embeddings[0])

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


print("FAISS vector database created!")
print("Number of vectors:", index.ntotal)
# Save FAISS index
faiss.write_index(index, "vectorstore/regulations.index")

print("FAISS index saved!")
# Save chunk metadata
with open("vectorstore/chunks.pkl", "wb") as file:
    pickle.dump(chunks, file)

print("Chunk metadata saved!")
# Test retrieval

question = "What is the minimum attendance required?"

question_embedding = model.encode([question])

distances, indices = index.search(question_embedding, 3)

print("\nRelevant information:\n")

for i in indices[0]:

    print("Source:", chunks[i]["source"])
    print("Page:", chunks[i]["page"])
    print("Text:", chunks[i]["text"])
    print("-" * 80)