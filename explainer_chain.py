from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Define the LLM
llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=groq_api_key,
    temperature=0.3
)

# Prompt template with conversation context
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an academic assistant. Use the previous conversation to inform your response."),
    ("human", "{chat_history}\n\nUser: {topic}\nAI:")
])

# Memory-aware function to run the explainer chain
def run_explainer_chain(inputs):
    topic = inputs["topic"]

    # Format previous chat history as a plain dialogue string
    chat_history = "\n".join([
        f"User: {msg}" if role == "user" else f"AI: {msg}"
        for role, msg in st.session_state.get("chat_history", [])
    ])

    # Format the prompt with history and current topic
    prompt = prompt_template.format(chat_history=chat_history, topic=topic)

    # Get response from LLM
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else response

    return {"result": answer}

# Make it LangGraph-compatible
explainer_chain = RunnableLambda(run_explainer_chain)
