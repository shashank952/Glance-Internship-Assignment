"""
Glance — FastAPI backend + static frontend for fashion image retrieval.
"""

import os, sys, json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Add project root to path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from retriever.query_parser import parse_query
from retriever.search import GlanceSearch

app = FastAPI(title="Glance — Fashion Image Retrieval")

# Mount image directory for serving
IMAGE_DIR = BASE / "data" / "sample_images"
app.mount("/images", StaticFiles(directory=str(IMAGE_DIR)), name="images")

# Mount static files (frontend)
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global search instance (loaded once at startup)
_search: GlanceSearch | None = None


@app.on_event("startup")
def load_model():
    global _search
    print("[Glance] Loading search index + model…")
    _search = GlanceSearch()
    print("[Glance] Ready.")


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/search")
def search(
    q: str = Query(..., description="Natural language query"),
    top_k: int = Query(10, ge=1, le=50),
):
    if _search is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)

    parsed = parse_query(q)
    results = _search.search(q, top_k=top_k)

    # Add image URL to each result
    for r in results:
        r["image_url"] = f"/images/{r['filename']}"

    return {
        "query": q,
        "parsed": parsed,
        "results": results,
        "total": len(results),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
