import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
import os

# ----------- CONFIG --------------------
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_huggingface_token_here"  # Replace this with your real token
pdf_path = "pdfs/IntroPython.pdf"  # Make sure this file exists in your GitHub repo
# --------------------------------------

# Load PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Embeddings + Vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# LLM
llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":0.5, "max_length":512})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Streamlit UI
st.title("📄 PDF AI Agent (Text Only)")
user_question = st.text_input("Ask something from the PDF")

if st.button("Get Answer") and user_question:
    answer = qa_chain.run(user_question)
    st.success(answer)
