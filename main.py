import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
import os

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    # Step 1: Generate Embedding Model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Step 2: Create FAISS index using from_texts
    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    # Step 3: Save the FAISS index
    vector_store.save_local("faiss_index")


llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-Nemo-Instruct-2407",  # Use the correct model repo
    model_kwargs={"temperature": 0.7, "max_tokens": 512,"return_full_text": False} ,
     huggingfacehub_api_token='' #Your API token
     )
   


def user_input(user_question):
    # Load embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load FAISS Vector Store
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # Construct Prompt
    prompt = f"""
    You are an expert assistant providing detailed answers based on the provided context.
    Understand the context provided thoroughly and then answer.  
    Use the information from the context to answer the question thoroughly.  
    Try to answer in a structured format for every question.
    If the answer is not available in the context, respond with "The answer is not available in the provided context."

    Context:
    {docs}

    Question:
    {user_question}

    Answer:
    """

    # Get Response from Mistral Model
    response = llm.invoke(prompt)

    # Display Response
    result = response.content.strip() if hasattr(response, 'content') else response.strip()
    print(result)
    st.write( result)


import time

def main():
  
   
    st.set_page_config(
    page_title="Chat with PDF 🤖",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded")
    st.header("Chat with PDF 🤖")
# Apply custom styles
    st.markdown(
        """
        <style>
        .title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            color: #4CAF50;
        }
        .stTextInput>div>div>input {
            font-size: 16px;
            padding: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Input for user questions
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        with st.spinner("Generating answer..."):
            user_input(user_question)

    # Sidebar for PDF Upload
    with st.sidebar:
        st.title("Ask Anything from the PDF")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on Submit & Process", 
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            with st.status("Starting the process...", expanded=True) as status:
                time.sleep(1)
                st.write("📥 Step 1: Collecting Data from PDFs... ✅")
                raw_text = get_pdf_text(pdf_docs)

                time.sleep(1)
                st.write("🔍 Step 2: Splitting Text into Chunks... ✅")
                text_chunks = get_text_chunks(raw_text)

                time.sleep(1)
                st.write("🧠 Step 3: Creating Vector Embeddings... ✅")
                get_vector_store(text_chunks)

                status.update(label="🎉 Processing Completed! ✅", state="complete")
                st.success("All steps completed successfully! 🚀")
    st.markdown(
    """
    ---
    Made with ❤️ by Balaji | 2025
    """,
    unsafe_allow_html=True,
)
      

if __name__ == "__main__":
    main()
