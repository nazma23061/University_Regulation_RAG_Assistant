import streamlit as st
import faiss
import pickle
import re

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -------------------------------------------------
# Page setup
# -------------------------------------------------

st.set_page_config(
    page_title="University Regulation RAG Assistant",
    page_icon="🎓"
)

st.title("🎓 University Regulation RAG Assistant")

st.write(
    "Ask questions about university regulations, academic rules, "
    "examinations, attendance, and student guidelines."
)


# -------------------------------------------------
# Load FAISS
# -------------------------------------------------

index = faiss.read_index(
    "vectorstore/regulations.index"
)


# -------------------------------------------------
# Load metadata
# -------------------------------------------------

with open("vectorstore/chunks.pkl", "rb") as file:
    chunks = pickle.load(file)


# -------------------------------------------------
# Load embedding model
# -------------------------------------------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


model = load_embedding_model()


# -------------------------------------------------
# Load AI model
# -------------------------------------------------

@st.cache_resource
def load_ai_model():

    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-small"
    )

    generator_model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-small"
    )

    return tokenizer, generator_model


tokenizer, generator_model = load_ai_model()


# -------------------------------------------------
# Document filter
# -------------------------------------------------

sources = sorted(
    set(chunk["source"] for chunk in chunks)
)

selected_source = st.selectbox(
    "📄 Filter by document:",
    ["All Documents"] + sources
)


# -------------------------------------------------
# Question
# -------------------------------------------------

question = st.text_input(
    "Enter your question:"
)


# -------------------------------------------------
# RAG
# -------------------------------------------------

if question:

    # Create question embedding
    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )


    # Search FAISS
    distances, indices = index.search(
        question_embedding,
        10
    )


    # -------------------------------------------------
    # Metadata filtering
    # -------------------------------------------------

    if selected_source != "All Documents":

        filtered_indices = []

        for i in indices[0]:

            if chunks[i]["source"] == selected_source:

                filtered_indices.append(i)

        selected_indices = filtered_indices[:3]

    else:

        selected_indices = list(indices[0][:3])


    # -------------------------------------------------
    # No matching document
    # -------------------------------------------------

    if len(selected_indices) == 0:

        st.warning(
            "⚠️ Information not found in the selected document."
        )

        st.stop()


    # -------------------------------------------------
    # Relevance check
    # -------------------------------------------------

    best_distance = float(distances[0][0])

    # FAISS L2 distance becomes smaller for more similar text.
    # A large distance means the retrieved information
    # is probably unrelated to the question.

    if best_distance > 1.2:

        st.warning(
            "⚠️ Information not found in the uploaded documents."
        )

        st.stop()


    # -------------------------------------------------
    # Prepare retrieved information
    # -------------------------------------------------

    retrieved_text = ""

    for i in selected_indices:

        retrieved_text += (
            chunks[i]["text"] + "\n"
        )


    # -------------------------------------------------
    # Grounded prompt
    # -------------------------------------------------

    prompt = f"""
You are a university regulation assistant.

Answer the question using ONLY the information in the CONTEXT.

Do not use your own knowledge.

If the CONTEXT does not contain the answer, respond exactly:

Information not found in the uploaded documents.

CONTEXT:
{retrieved_text}

QUESTION:
{question}

ANSWER:
"""


    # -------------------------------------------------
    # Generate answer
    # -------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )


    outputs = generator_model.generate(
        **inputs,
        max_new_tokens=100
    )


    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


    # -------------------------------------------------
    # Display answer
    # -------------------------------------------------

    st.subheader("💡 Answer")

    st.write(answer)


    # -------------------------------------------------
    # Display sources
    # -------------------------------------------------

    st.subheader("📚 Sources")

    shown = set()

    for i in selected_indices:

        source = chunks[i]["source"]

        page = chunks[i]["page"]

        key = (source, page)

        if key not in shown:

            st.write(
                f"📄 **{source}** — Page **{page}**"
            )

            shown.add(key)