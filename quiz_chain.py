from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
import json
import os
from dotenv import load_dotenv

# Load environment variables (API key)
load_dotenv()

# Initialize the LLM
llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=os.getenv("GROQ_API_KEY")
)

# Define the prompt template for generating a quiz
prompt = PromptTemplate.from_template(
"""
You are a quiz generator.

Your task is to generate exactly 10 multiple-choice questions based on the following document content. 
Each question must have 4 answer options and only one correct answer. 

Return the result strictly in **valid JSON array** format like this:

[
  {{
    "question": "What is select() used for?",
    "options": ["To read input", "To monitor multiple descriptors", "To send signals"],
    "answer": "To monitor multiple descriptors"
  }},
  ...
]

Do not include any explanations or markdown. Just the JSON array only.

Content:
{content}
"""
)

# Create the quiz generation chain
quiz_chain = LLMChain(llm=llm, prompt=prompt)

# Define a function to run the quiz chain and parse the JSON output
def run_quiz_chain(content: str):
    try:
        raw_output = quiz_chain.run(content)
        print("RAW OUTPUT FROM LLM:", raw_output)  # For debugging
        parsed_output = json.loads(raw_output)
        return parsed_output
    except Exception as e:
        return [{
            "question": "Quiz generation failed.",
            "options": [],
            "answer": str(e)
        }]

