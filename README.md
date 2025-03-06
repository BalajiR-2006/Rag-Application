# 📄 Chat with PDF 🤖 - RAG Application

## 🚀 Overview
This is a **Retrieval-Augmented Generation (RAG) based chatbot** that allows users to **ask questions** about uploaded **PDF documents**. The application:
- Extracts text from PDFs 💑
- Splits text into manageable chunks 🔍
- Creates vector embeddings using Google's Generative AI 🧠
- Stores embeddings in FAISS for efficient retrieval ⚡
- Uses a **Mistral model** for answering user queries intelligently 🤖

## 🎨 Application Screenshots
### **1️⃣ Uploading and Processing PDFs**
![Uploading PDFs](Screenshot%202025-03-06%20172649.png)

### **2️⃣ Asking Questions from the PDF**
![Asking Questions](Screenshot%202025-03-06%20172606.png)

---

## 🛠️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-repo/chat-with-pdf.git
cd chat-with-pdf
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up API Keys
Create a `.env` file in the root directory and add your API keys:
```bash
GOOGLE_API_KEY=your_google_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key
```

### 4️⃣ Run the Application
```bash
streamlit run app.py
```

---

## 🎯 Usage
1. Upload one or multiple **PDF files**.
2. Click **Submit & Process** to extract, chunk, and embed the text.
3. Type a **question** in the input box.
4. The AI will retrieve the most relevant content and generate an **accurate answer**.

---

## 🛠️ Technologies Used
- **Python** 🐍
- **Streamlit** - For UI
- **PyPDF2** - For extracting text from PDFs
- **LangChain** - For chunking, retrieval, and querying
- **FAISS** - For storing and searching embeddings
- **Google Generative AI (Gemini)** - For embeddings
- **Mistral (HuggingFace Hub)** - For LLM-based response generation

---

## 🔥 Features
✅ Upload and process PDFs  
✅ Query PDFs using natural language  
✅ Fast and accurate responses  
✅ Uses **retrieval-based augmented generation (RAG)**  
✅ Supports **multiple document processing**  

---

## 📌 Future Improvements
- Support for **more LLMs** like GPT-4 or Claude  
- Better **UI enhancements**  
- Option to **download** responses  
- **Multi-modal support** (Images & PDFs)  

---

## 🤝 Contributing
Feel free to **fork** the repo, raise **issues**, and submit **PRs**!

---

## 🐟 License
This project is **MIT Licensed**.

---

