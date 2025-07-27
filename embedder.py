# utils/embedder.py

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

def store_embeddings(chunks, persist_directory="db"):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb

def load_embeddings(persist_directory="db"):
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
