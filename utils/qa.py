# utils/qa.py

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def get_qa_chain(vectordb):
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return chain
    
