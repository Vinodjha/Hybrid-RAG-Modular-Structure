from fastapi import FastAPI
from app.api.routes_health import router as health_router
from app.api.routes_index import router as index_router
from app.api.routes_query import router as query_router

app = FastAPI(title="RAG Retrieval Service", version="1.0")
app.include_router(health_router)
app.include_router(index_router)
app.include_router(query_router)
