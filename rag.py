import os
import shutil
from dotenv import load_dotenv
load_dotenv()

import chromadb

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


CHROMA_DIR = "chroma_db"


def load_url(url: str):
    loader = UnstructuredURLLoader(
        urls=[url],
        languages=["en", "te", "hi", "ta"]
    )
    return loader.load()


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)


def create_vector_db(chunks):

    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    client = chromadb.Client(
        chromadb.config.Settings(
            persist_directory=CHROMA_DIR,
            anonymized_telemetry=False
        )
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name="url_rag"
    )

    return vectordb


def format_docs(docs):
    """Convert retrieved docs to plain text context"""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(db):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=500,
        temperature=0
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful URL reader.

STRICT RULES:
- Use ONLY the context below
- Do NOT hallucinate
- DEFAULT language is ENGLISH
- Change the answer language ONLY if the CURRENT question explicitly asks for it
- Do NOT remember language preference from previous questions
- Supported languages: English, Telugu, Tamil, Hindi
- If the user says "telugu", "తెలుగు", "తెలుగులో", answer in Telugu
- If the user says "tamil", "தமிழ்", answer in Tamil
- If the user says "hindi", "हिंदी", answer in Hindi
- Otherwise, ALWAYS answer in English

Context:
{context}

User Question:
{question}

Answer (follow the language rules strictly):
"""
    )

    retriever = db.as_retriever(search_kwargs={"k": 4})

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return chain
