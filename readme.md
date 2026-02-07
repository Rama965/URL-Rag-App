# URL Loader RAG App 🔗🤖

A **Retrieval-Augmented Generation (RAG)** application that loads content from URLs, retrieves relevant information, and generates accurate, context-aware answers using LLMs.

Built with **Python, Streamlit, and LangChain**, and deployed on **Streamlit Community Cloud**.

---

## 🚀 Features

- Load content directly from URLs  
- Text chunking and vector embedding creation  
- Context retrieval using RAG architecture  
- Intelligent response generation using LLMs  
- Secure API key handling with environment variables  
- Simple and interactive Streamlit UI  

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- LangChain  
- Vector Embeddings  
- LLMs  
- Environment Variables (.env)

---

## 📂 Project Structure

-URL-Rag-App/
├── app.py
├── rag.py
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md



---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Rama965/URL-Rag-App.git
cd URL-Rag-App

2. Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Configure Environment Variables

Create a .env file:

GROQ_API_KEY=your_api_key_here


▶️ Run the App
streamlit run app.py