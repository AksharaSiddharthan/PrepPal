# main.py

import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

from utils.loader import load_pdf
from utils.splitter import split_docs
from utils.embedder import store_embeddings, load_embeddings
from utils.qa import get_qa_chain

# Load environment variables (for OpenAI API Key)
load_dotenv()

st.set_page_config(page_title="Prepal - AI Study Buddy", layout="wide")
st.title("📘 Prepal — Understand Your Notes with AI")

# PDF Upload
uploaded_file = st.file_uploader("📄 Upload your study material (PDF)", type="pdf")

if uploaded_file:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name

    st.success("PDF uploaded! Now processing...")

    # Load and chunk the PDF
    docs = load_pdf(temp_pdf_path)
    chunks = split_docs(docs)

    # Embed and store in ChromaDB
    vectordb = store_embeddings(chunks)

    # Create QA chain and store in session
    qa_chain = get_qa_chain(vectordb)
    st.session_state.qa_chain = qa_chain

    st.success("✅ PDF processed! You can now ask questions.")

# Question input
if "qa_chain" in st.session_state:
    question = st.text_input("❓ Ask a question about your notes:")
    
    if question:
        with st.spinner("Thinking..."):
            answer = st.session_state.qa_chain.run(question)
        st.markdown(f"**Answer:** {answer}")
