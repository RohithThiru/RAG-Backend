from fastapi import APIRouter, UploadFile, File
import shutil
import os

from app.rag import ingest_pdf, ask_question

router = APIRouter()


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{file.filename}"

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ingest_pdf(path, file.filename)

    return {"status": "uploaded", "filename": file.filename}


@router.get("/documents")
def list_documents():
    return os.listdir("uploads")


@router.post("/ask")
def ask(payload: dict):
    return ask_question(payload["question"])
