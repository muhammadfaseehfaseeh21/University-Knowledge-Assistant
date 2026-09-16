import os
import glob

import streamlit as st
import fitz
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from groq import Groq


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="University Knowledge Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 University Knowledge Assistant")
st.write(
    "Ask questions about university admissions, fees, scholarships, "
    "examinations, academic calendar, and student policies."
)


# --------------------------------------------------
# GROQ API
# --------------------------------------------------

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY is not configured.")
    st.stop()

client = Groq(api_key=groq_api_key)


# --------------------------------------------------
# DOCUMENT FOLDER
# --------------------------------------------------

DOCUMENT_FOLDER = "documents"


# --------------------------------------------------
# LOAD PDF DOCUMENTS
# --------------------------------------------------

def load_documents():
    documents = []

    pdf_files = glob.glob(
        os.path.join(DOCUMENT_FOLDER, "*.pdf")
    )

    for pdf_file in pdf_files:

        file_name = os.path.basename(pdf_file)

        pdf = fitz.open(pdf_file)

        for page_number, page in enumerate(pdf):

            text = page.get_text()

            if text.strip():

                documents.append({
                    "text": text,
                    "source": file_name,
                    "page": page_number + 1
                })

        pdf.close()

    return documents


# --------------------------------------------------
# CREATE TEXT CHUNKS
# --------------------------------------------------

def create_chunks(documents, chunk_size=800):

    chunks = []

    for document in documents:

        text = document["text"]

        words = text.split()

        for i in range(0, len(words), chunk_size):

            chunk_text = " ".join(
                words[i:i + chunk_size]
            )

            if chunk_text.strip():

                chunks.append({
                    "text": chunk_text,
                    "source": document["source"],
                    "page": document["page"]
                })

    return chunks


# --------------------------------------------------
# CREATE EMBEDDINGS
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# --------------------------------------------------
# BUILD FAISS DATABASE
# --------------------------------------------------

@st.cache_resource
def build_vector_database():

    documents = load_documents()

    if not documents:
        return None, [], 0, 0

    chunks = create_chunks(documents)

    embedding_model = load_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index, chunks, len(documents), len(chunks)


# --------------------------------------------------
# BUILD KNOWLEDGE BASE
# --------------------------------------------------

with st.spinner("Loading university knowledge base..."):

    index, chunks, document_count, chunk_count = (
        build_vector_database()
    )


# --------------------------------------------------
# CHECK DOCUMENTS
# --------------------------------------------------

if index is None:

    st.warning(
        "No PDF documents were found in the documents folder."
    )

    st.info(
        "Please add your university PDF files inside the "
        "'documents' folder."
    )

    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("📚 Knowledge Base")

st.sidebar.write(
    f"📄 Documents: {document_count}"
)

st.sidebar.write(
    f"🧩 Text chunks: {chunk_count}"
)

st.sidebar.markdown("---")

st.sidebar.subheader("Available Documents")

for chunk in chunks:

    pass


available_files = sorted(
    list(set(
        chunk["source"]
        for chunk in chunks
    ))
)

for file_name in available_files:

    st.sidebar.write(
        f"📄 {file_name}"
    )


# --------------------------------------------------
# SEARCH KNOWLEDGE BASE
# --------------------------------------------------

def search_documents(question, top_k=5):

    embedding_model = load_embedding_model()

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    question_embedding = question_embedding.astype(
        "float32"
    )

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx < len(chunks):

            results.append({
                "text": chunks[idx]["text"],
                "source": chunks[idx]["source"],
                "page": chunks[idx]["page"],
                "distance": float(distance)
            })

    return results


# --------------------------------------------------
# AI CHATBOT
# --------------------------------------------------

def generate_answer(question, results):

    context_parts = []

    for result in results:

        context_parts.append(
            f"""
Source: {result['source']}
Page: {result['page']}

Content:
{result['text']}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a University Knowledge Assistant.

Answer the student's question using ONLY the
information provided in the university documents below.

If the answer cannot be found in the documents,
clearly say:

"I could not find this information in the provided
university documents."

Do not invent university policies, fees, dates,
rules, or requirements.

Always try to mention the relevant document source
when answering.

University Documents:

{context}

Student Question:

{question}


    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful university "
                    "knowledge assistant."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2
    )

    return response.choices[0].message.content


# --------------------------------------------------
# CHAT INTERFACE
# --------------------------------------------------

st.subheader("💬 Ask the University Assistant")

question = st.chat_input(
    "Ask about admissions, fees, scholarships, exams, etc."
)


if question:

    # Show student question
    with st.chat_message("user"):

        st.write(question)

    # Search relevant documents
    with st.spinner("Searching university documents..."):

        results = search_documents(
            question,
            top_k=5
        )

    # Generate AI answer
    with st.spinner("Generating answer..."):

        answer = generate_answer(
            question,
            results
        )

    # Show answer
    with st.chat_message("assistant"):

        st.write(answer)

        st.markdown("### 📚 Sources")

        shown_sources = set()

        for result in results:

            source_key = (
                result["source"],
                result["page"]
            )

            if source_key not in shown_sources:

                st.write(
                    f"📄 {result['source']} — "
                    f"Page {result['page']}"
                )

                shown_sources.add(source_key)
