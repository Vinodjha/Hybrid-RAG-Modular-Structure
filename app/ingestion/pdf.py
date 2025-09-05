#Ingesting PDFs docs

from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from app.core.settings import Settings
import os

def chunk_pdf(path:str) -> List[Dict[str, Any]]:
    """Load and chunk a PDF file into smaller text chunks."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    loader = PyPDFLoader(path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Settings().CHUNK_SIZE,
        chunk_overlap=Settings().CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)
    items = []
    for d in chunks:
        items.append({
            "text": d.page_content,
            "page": d.metadata.get("page", -1),
            "source": d.metadata.get("source", os.path.basename(path))
        })
    return items

    