from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.chains import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
import streamlit as st
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Define the LLM
llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=groq_api_key,
    temperature=0.3
)

# Set up vector database and retriever
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
retriever = vectordb.as_retriever()

# Prompt template that incorporates previous chat history
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an academic assistant. Use the following chat history and retrieved documents to answer the user's new question."),
    ("human", "{history}\n\nQuestion: {input}")
])

# Memory-aware retriever chain function
def run_retriever_chain(inputs):
    query = inputs["query"]

    # Create a ConversationalRetrievalChain with source documents returned
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # Send the query with previous chat history
    response = qa({
        "question": query,
        "chat_history": st.session_state.chat_history  # Stored in session
    })

    answer = response["answer"]
    sources = response.get("source_documents", [])

    # Format source information for display
    unique_sources = set()
    for doc in sources:
        filename = os.path.basename(doc.metadata.get("source", "Unknown"))
        page_num = doc.metadata.get("page", "N/A")
        unique_sources.add(f"📄 **{filename}** (Page {page_num})")

    source_info = "\n\n" + "\n".join(unique_sources)

    # Combine answer with source reference
    full_response = f"{answer}\n\n{source_info.strip()}"

    return {"result": full_response}

# Wrap as a LangGraph-compatible runnable
retriever_chain = RunnableLambda(run_retriever_chain)
