import os
import tempfile
import speech_recognition as sr
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter

# --- Config ---
PDF_FOLDER = "pdfs"
INDEX_FOLDER = "faiss_index_local"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- Setup Streamlit ---
st.set_page_config(page_title="PDF Voice QA Agent", layout="centered")
st.title("🧠 Voice-Powered PDF QA Agent")

# --- Step 1: Load PDFs and Build Vector Store ---
@st.cache_resource(show_spinner="Loading documents and building vector store...")
def load_or_build_vectorstore():
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(INDEX_FOLDER):
        return FAISS.load_local(INDEX_FOLDER, embedding, allow_dangerous_deserialization=True)

    all_docs = []
    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(PDF_FOLDER, file))
            all_docs.extend(loader.load())

    docs = text_splitter.split_documents(all_docs)
    vectorstore = FAISS.from_documents(docs, embedding)
    vectorstore.save_local(INDEX_FOLDER)
    return vectorstore

vectorstore = load_or_build_vectorstore()
retriever = vectorstore.as_retriever()

# --- Step 2: Setup LLM ---
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    huggingfacehub_api_token=HF_TOKEN,
    model_kwargs={"temperature": 0.5, "max_length": 256},
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# --- Step 3: Get Voice Input ---
def transcribe_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening... Speak now")
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        st.success(f"✅ You said: {query}")
        return query
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return ""

# --- Step 4: User Interaction ---
if st.button("🎤 Ask with Voice"):
    query = transcribe_speech()
    if query:
        with st.spinner("Answering..."):
            result = qa_chain.run(query)
            st.success("🧠 Answer:")
            st.write(result)

# --- Optional text input ---
st.markdown("---")
st.write("Or type your question:")
text_query = st.text_input("Question")
if text_query:
    with st.spinner("Answering..."):
        result = qa_chain.run(text_query)
        st.success("🧠 Answer:")
        st.write(result)
