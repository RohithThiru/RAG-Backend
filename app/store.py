from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from app.config import CHROMA_DIR, COLLECTION_NAME

embeddings = OpenAIEmbeddings()

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)
