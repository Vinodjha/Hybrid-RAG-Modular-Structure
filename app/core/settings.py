import os
from dataclasses import dataclass
from dotenv import load_dotenv

# 👇 this line must run first
load_dotenv()
#immutable settings with dataclass
@dataclass(frozen=True)

class Settings:
    EMB_MODEL_NAME: str = os.getenv("EMB_MODEL","sentence-transformers/all-MiniLM-L6-v2")
    RERANK_MODEL_NAME:str = os.getenv("RERANK_MODEL","cross-encoder/ms-marco-MiniLM-L-6-v2")    
    LLM_MODEL:str = os.getenv("LLM_MODEL","llama-3.3-70b-versatile")

    CHUNK_SIZE:int = int(os.getenv("CHUNK_SIZE","500"))
    CHUNK_OVERLAP:int = int(os.getenv("CHUNK_OVERLAP","100"))

    K_FETCH:int = int(os.getenv("K_FETCH","25"))
    K_MMR:int = int(os.getenv("K_MMR","8"))
    M_FINAL:int = int(os.getenv("M_FINAL","4"))
    TOKEN_BUDGET:int = int(os.getenv("TOKEN_BUDGET","1200"))

    DATA_DIR:str = os.getenv("DATA_DIR","./data")
    INDEX_DIR:str = os.getenv("INDEX_DIR","./index")

    SYSTEM_PROMPT:str = os.getenv("SYSTEM_PROMPT","You are a precise crypto research assistant. Cite sources with [pX]. Be concise.")

    GROQ_API_KEY:str = os.getenv("GROQ_API_KEY","")

settings = Settings()

