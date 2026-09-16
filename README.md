# University-Knowledge-Assistant# 🎓 University Knowledge Assistant

A RAG-based University Knowledge Assistant built with:

- Python
- Streamlit
- FAISS
- Sentence Transformers
- Groq API
- PyMuPDF

## Features

- 📚 Multiple PDF knowledge base
- 🤖 AI chatbot
- 🔎 Semantic document search
- 🎓 Student Handbook
- 💰 Fee Policy
- 📝 Admission Guidelines
- 💵 Scholarship Policy
- 📊 Examination Rules
- 📅 Academic Calendar
- 📄 Source document and page references

## Project Structure

university-knowledge-assistant/

├── app.py
├── requirements.txt
├── README.md
│
└── documents/
    ├── student_handbook.pdf
    ├── fee_policy.pdf
    ├── admission_guidelines.pdf
    ├── scholarship_policy.pdf
    ├── examination_rules.pdf
    └── academic_calendar.pdf

## Setup

Install the requirements:

pip install -r requirements.txt

## Groq API Key

Set your Groq API key as:

GROQ_API_KEY

## Run

streamlit run app.py

## How RAG Works

1. PDF documents are loaded from the documents folder.
2. Text is extracted from the PDFs.
3. Text is divided into chunks.
4. Sentence Transformer creates embeddings.
5. FAISS stores the embeddings.
6. Student asks a question.
7. FAISS finds relevant document chunks.
8. The relevant chunks are sent to the Groq LLM.
9. The AI generates an answer using the retrieved information.

## Supported Knowledge

The assistant can answer questions related to:

- Admissions
- Registration
- Fees
- Scholarships
- Examination rules
- Academic calendar
- Student handbook
- University policies
