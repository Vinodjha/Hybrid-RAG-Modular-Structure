from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os # 

from app.api.routes_health import router as health_router
from app.api.routes_index import router as index_router
from app.api.routes_query import router as query_router
from app.api.routes_wipe import router as wipe_router 

# Determine the absolute path of the directory where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the 'static' directory
static_dir = os.path.join(base_dir, "static")

# Mount a directory to serve static files (like CSS, JS, or images)
app = FastAPI(title="RAG Retrieval Service", version="1.0")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory=static_dir)

app.include_router(health_router)
app.include_router(index_router)
app.include_router(query_router)
app.include_router(wipe_router) 

@app.get("/", response_class=HTMLResponse, tags=["ui"])
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

print("ROUTES:", [(r.path, getattr(r, "methods", None)) for r in app.router.routes])