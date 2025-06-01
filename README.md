# 🎓 AI Academic Assistant

This is a modular, LangChain + LangGraph powered Streamlit application that assists students with academic content. It provides topic explanations, document-based question answering, summarization, and auto-generated quizzes.

---

## 🚀 Features

- **Topic Explanation**: Ask questions and get detailed academic explanations.
- **Document Q&A (RAG)**: Upload PDFs and get answers grounded in your documents.
- **Summarization**: Generate concise summaries from long documents.
- **Quiz Generator**: Create multiple-choice questions from uploaded content.
- **Styled UI**: Clean and modern chat-style interface with Streamlit + CSS.

---

## 🧠 Technologies Used

- [LangChain](https://www.langchain.com/) – LLM framework  
- [LangGraph](https://www.langgraph.dev/) – Flow-based routing logic  
- [Groq LLM](https://console.groq.com/) – LLaMA-4 model host  
- [ChromaDB](https://www.trychroma.com/) – Vector database  
- [Streamlit](https://streamlit.io/) – UI framework  
- [Hugging Face Transformers](https://huggingface.co/) – Sentence embedding  

---

## 📂 Project Structure

```bash
AcademicAssistant/
├── langgraph_app.py         # Main Streamlit interface
├── langgraph_chain.py       # LangGraph flow controller
├── explainer_chain.py       # Chain for topic explanations
├── retriever_chain.py       # Chain for document Q&A
├── summary_chain.py         # Summarization tool
├── quiz_chain.py            # Quiz generation tool
├── ingestion.ipynb          # PDF loader and text chunking
├── style.css                # UI styling for Streamlit
├── .env.example             # API key config (example only)
├── requirements.txt         # Dependencies list
├── .gitignore               # Files to exclude from Git
├── chroma_db/               # Persistent vector store (auto-generated)
└── data/
    └── .gitkeep             # Keeps data folder tracked
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-academic-assistant.git
cd ai-academic-assistant
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Your API Key

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_api_key_here
```

### 4. Run the App

```bash
streamlit run langgraph_app.py
```

---

## 🧪 Example Use

- Upload a PDF (like a lecture slide)  
- Ask a question: “What is backpropagation?”  
- Generate summary  
- Click “Generate Quiz” and solve the test  

---

## 📄 Project Report

For full technical details, architecture, modules, experiments, and evaluations, see:

➡️ [Project_Report_MelisCan.pdf](./Project_Report_MelisCan.pdf)

---

## 🙋 Author

**Melis Can**  
[GitHub Profile](https://github.com/meliscann)

---

## 🛡️ License

This project is for educational purposes and not licensed for commercial use.
