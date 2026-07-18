# AGENTS.md — Glance Fashion Image Retrieval

**Current state**: Post-implementation. Full pipeline at `app/` (FastAPI), `indexer/`, `retriever/`. Docs in `docs/`.

## One constraint to get right

Vanilla CLIP + FAISS is **explicitly rejected** by the assignment. Every solution must beat CLIP on compositional queries (e.g. "red shirt blue pants" must NOT retrieve "blue shirt red pants"). See [PRD.md §5](docs/PRD.md#51-part-a--the-indexer) for how — structured attribute vectors + multi-stage reranking solves this.

## Design source of truth

[PRD.md](docs/PRD.md) — full architecture, indexing pipeline, retriever, evaluation plan, scalability discussion. Read it before writing any code. The PDF (`docs/Glance ML Internship Assignment.pdf`) contains the original assignment prompt.

## Expected layout (agents: place new files here)

```
data/            images + metadata.csv (category, color, pseudo-location, pseudo-style)
indexer/         pseudo-label gen, embedding, vector index build
retriever/       query parsing, vector search + rerank, evaluation
models/          model wrappers (CLIP/SigLIP, etc.)
evaluation/      metrics, ablation, failure analysis
app/             Web UI (FastAPI backend + static HTML frontend)
docs/            report, PRD, limitations
```

## Python toolchain

WIP — no deps installed yet. When setting up:
- Tool: `uv` (preferred) or `pip`
- Key deps: `faiss-cpu`, `transformers`, `torch`, `pandas`
- Full stack: [PRD.md §6](docs/PRD.md#6-tech-stack)

## Key commands

```bash
# Index pipeline (run in order)
python indexer/generate_metadata.py
python indexer/generate_pseudo_labels.py
python indexer/build_embeddings.py
python indexer/build_vector_index.py

# Search
python retriever/search.py "A red tie and a white shirt in a formal setting"

# Evaluate (all 5 queries × 5 pipeline variants)
python retriever/evaluate.py

# Web UI
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Architecture

See `images/workflow.png` for the system diagram. No LLM in the pipeline — query parsing is keyword-based.

## Evaluation queries (ground truth for any impl)

| # | Query | What it tests |
|---|---|---|
| 1 | "A person in a bright yellow raincoat." | Color + garment binding |
| 2 | "Professional business attire inside a modern office." | Location + style inference |
| 3 | "Someone wearing a blue shirt sitting on a park bench." | Multi-attribute composition |
| 4 | "Casual weekend outfit for a city walk." | Style-only (no explicit garment) |
| 5 | "A red tie and a white shirt in a formal setting." | **Compositional** (two colors, two garments — vanilla CLIP swaps them here) |

## Reference files

| File | Role |
|---|---|
| `docs/PRD.md` | Architecture, pipeline design, evaluation plan, scalability |
| `docs/report.md` | Evaluation results and failure analysis |
| `docs/LIMITATIONS.md` | Known limitations |
| `docs/Glance ML Internship Assignment.pdf` | Original assignment prompt |
| `images/workflow.png` | System architecture diagram |
