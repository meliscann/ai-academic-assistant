import streamlit as st
from langgraph_chain import langgraph_chain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from summary_chain import run_summary_chain
from quiz_chain import run_quiz_chain
import shutil
import os

# Set up the page configuration for Streamlit
st.set_page_config(page_title="AI Academic Assistant (LangGraph)", layout="centered")

# Load external CSS for scroll behavior and layout styling
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize chat history if not already in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Render the header
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.title("🧠🌟 AI Academic Assistant")
st.markdown("Choose a mode and ask your question. The assistant will either explain the topic or use your documents to answer.")

# Mode selection: general explanation or document-based Q&A
mode = st.radio("Select mode:", ["Topic Explanation", "Document Q&A"])
disabled_state = mode != "Document Q&A"

if mode == "Topic Explanation":
    st.info("💡 In this mode, the assistant uses its academic knowledge to explain topics and answer questions, without relying on uploaded documents.")
else:
    st.warning("📄 In this mode, the assistant will only use the content from your uploaded documents to generate answers.")

# Start scrollable chat area
st.markdown('<div class="chat-scroll-area">', unsafe_allow_html=True)

# Display previous chat messages
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(f'<div class="chat-bubble user">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'''
            <div class="chat-bubble assistant">
                <img src="https://cdn-icons-png.flaticon.com/512/4712/4712105.png" class="chat-icon" />
                <div class="assistant-message">{message}</div>
            </div>
        ''', unsafe_allow_html=True)

# Close chat area container
st.markdown('</div>', unsafe_allow_html=True)

# Upload PDF document (only enabled in Document Q&A mode)
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], disabled=disabled_state)
if uploaded_file is not None:
    file_path = f"./data/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ File uploaded successfully!")

# Display uploaded documents
st.markdown("### 📂 Uploaded Documents")
selected_file = None
if os.path.exists("./data"):
    files = [f for f in os.listdir("./data") if f.endswith(".pdf")]
    if files:
        selected_file = st.selectbox("Select a document to use:", files)
        for file in files:
            col1, col2 = st.columns([6, 1])
            col1.write(f"📄 {file}")
            if col2.button("🗑️ Delete", key=file):
                os.remove(os.path.join("./data", file))
                st.success(f"{file} has been deleted.")
                st.rerun()
    else:
        st.info("No uploaded documents found.")
else:
    st.info("No documents directory found.")

# Educational tools (only active in Document Q&A mode)
st.markdown("### 📘️ Educational Tools")

if mode == "Document Q&A" and selected_file:
    file_path = os.path.join("./data", selected_file)
    if os.path.exists(file_path):
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        for i, page in enumerate(pages):
            page.metadata["source"] = selected_file
            page.metadata["page"] = i + 1

        embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
        vectordb.add_documents(pages)

        # Generate a summary of the document
        if st.button("📖 Summary of the Document"):
            with st.spinner("Generating summary..."):
                summary = run_summary_chain(pages)
                st.markdown(summary)

        # Generate a quiz based on the document content
        if st.button("📋 Generate Quiz"):
            with st.spinner("Generating quiz..."):
                content = "\n".join([p.page_content for p in pages])
                quiz = run_quiz_chain(content)
                st.session_state.generated_quiz = quiz
                st.session_state.quiz_data = quiz

# Quiz solving interface (only available in Document Q&A mode)
if mode == "Document Q&A" and "quiz_data" in st.session_state:
    st.markdown("### 📝 Solve the Quiz!")
    user_answers = {}

    with st.form("quiz_form"):
        for i, item in enumerate(st.session_state.quiz_data):
            st.write(f"**Q{i+1}: {item['question']}**")
            user_answers[i] = st.radio(
                label="",
                options=item["options"],
                key=f"quiz_q{i}",
                index=None
            )
            st.markdown("---")

        submitted = st.form_submit_button("Submit Answers")

    if submitted:
        score = 0
        total = len(st.session_state.quiz_data)
        st.markdown("### 📊 Results:")
        for i, item in enumerate(st.session_state.quiz_data):
            st.write(f"**Q{i+1}: {item['question']}**")
            user_answer = user_answers[i]
            correct_answer = item["answer"]
            if user_answer == correct_answer:
                st.markdown(f"- Your answer: **{user_answer}** ✅")
                score += 1
            else:
                st.markdown(f"- Your answer: **{user_answer}** ❌")
                st.markdown(f"- Correct answer: **{correct_answer}**")
            st.markdown("---")
        st.success(f"🎉 You got {score} out of {total} correct!")

# User input section
st.markdown('<div class="chat-input">', unsafe_allow_html=True)
user_input = st.text_input("Enter your question or topic:")
if st.button("🧐 Ask"):
    if user_input.strip():
        st.session_state.chat_history.append(("user", user_input))
        with st.spinner("Thinking..."):
            state = {"query": user_input, "mode": "qa" if mode == "Document Q&A" else "explain"}
            result = langgraph_chain.invoke(state)
        if "result" in result:
            st.session_state.chat_history.append(("ai", result["result"]))
        st.rerun()
st.markdown('</div></div>', unsafe_allow_html=True) 

# Button to clear chat history
if st.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

# Button to reset the vector database (only active in Document Q&A mode)
if st.button("🔄 Reset Database", disabled=disabled_state):
    try:
        shutil.rmtree("./chroma_db", ignore_errors=True)
        st.success("✅ Database has been reset successfully. You can now re-select a document.")
    except Exception as e:
        st.error(f"❌ Error deleting database: {str(e)}")
