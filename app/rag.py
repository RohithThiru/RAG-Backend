import uuid
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


# -------------------------------
# LLM (temperature 0 = no guessing)
# -------------------------------
def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

# -------------------------------
# Vector Store
# -------------------------------
vectorstore = Chroma(
    persist_directory="./chroma",
    embedding_function=OpenAIEmbeddings()
)

# -------------------------------
# Text Splitter
# -------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " "]
)


# =========================================================
# INGEST PDF (UNCHANGED)
# =========================================================
def ingest_pdf(file_path: str, filename: str):
    reader = PdfReader(file_path)

    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""

    docs = [
        Document(
            page_content=chunk,
            metadata={"source": filename}
        )
        for chunk in text_splitter.split_text(full_text)
    ]

    vectorstore.add_documents(docs)


# =========================================================
# LIST DOCUMENTS (OPTIONAL / UNCHANGED)
# =========================================================
def list_documents():
    results = vectorstore.get(include=["metadatas"])

    filenames = set()
    for meta in results["metadatas"]:
        filenames.add(meta.get("source"))

    return [{"filename": name} for name in filenames]


# =========================================================
# ASK QUESTION (STRICT RAG – NO HALLUCINATION)
# =========================================================

def ask_question(question: str):
    results = vectorstore.similarity_search_with_score(
        question,
        k=3
    )

    # results = [(Document, score)]

    # Chroma uses cosine distance → lower is better
    # Typical good threshold: 0.2 – 0.35
    MAX_DISTANCE = 0.90

    filtered_docs = [
        doc for doc, score in results if score <= MAX_DISTANCE
    ]

    llm = get_llm()


    context = "\n\n".join(d.page_content for d in filtered_docs)

    response = llm.invoke(
        f"""
        You are a helpful assistant that answers questions based on the provided context.
                {context}
                Question:
                {question}
        """
    )

    return {
        "answer": response.content,
        "sources": [d.metadata for d in filtered_docs]
    }
