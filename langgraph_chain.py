import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from retriever_chain import retriever_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda

# Define the state structure for the LangGraph
class GraphState(TypedDict):
    query: str
    result: str
    mode: Literal["explain", "qa"]

# Initialize memory to track conversation history
memory = ConversationBufferMemory(return_messages=True, input_key="input", memory_key="history")

# Set up the LLM
llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

# Create a prompt that uses chat history for context-aware explanation
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an academic assistant. Maintain memory of the user's previous questions and your responses during this session, and provide context-aware explanations."),
    ("human", "{input}"),
])

# Create an explainer chain with memory support
explainer_llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Define the explainer function
def run_explainer(state: GraphState) -> GraphState:
    topic = state["query"]
    answer = explainer_llm_chain.run({"input": topic})
    return {"query": topic, "result": answer, "mode": "explain"}

# Define the retriever function
def run_retriever(state: GraphState) -> GraphState:
    query = state["query"]
    response = retriever_chain.invoke({"query": query})
    answer = response["result"]
    return {"query": query, "result": answer, "mode": "qa"}

# Define a routing function to determine the entry node
def route(state: GraphState):
    return "retriever" if state["mode"] == "qa" else "explainer"

# Build the LangGraph
builder = StateGraph(GraphState)
builder.add_node("explainer", run_explainer)
builder.add_node("retriever", run_retriever)
builder.set_conditional_entry_point(route)
builder.add_edge("explainer", END)
builder.add_edge("retriever", END)

# Compile the LangGraph chain
langgraph_chain = builder.compile()
