from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

# Load environment variables (API key)
load_dotenv()

# Initialize the LLM
llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct", 
    api_key=os.getenv("GROQ_API_KEY")
)

# Create the summarization chain using the "map_reduce" strategy
chain = load_summarize_chain(llm, chain_type="map_reduce")

# Define the function to run the summarization chain
def run_summary_chain(pages: list[Document]) -> str:

    # If the document is too long, split it into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    # Run the summarization chain
    result = chain.run(docs)
    
    return result
