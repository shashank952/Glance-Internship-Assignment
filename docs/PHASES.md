# PHASES.md — Implementation Phases

## Phase 0: Environment Setup
**Files**: `requirements.txt`, directory structure
**Actions**:
- Create directory tree: `indexer/`, `retriever/`, `evaluation/`, `data/`, `models/`, `app/`, `docs/`
- Write `requirements.txt` (faiss-cpu, transformers, torch, pandas, pillow, scikit-learn, numpy)
- Install dependencies via `uv` or `pip`
- Verify Python + GPU/CUDA availability

## Phase 1: Metadata Generation — Category + Color
**File**: `indexer/generate_metadata.py`
**Actions**:
- Fashion CLIP zero-shot classification → 20 garment categories
- KMeans (k=3) in LAB on center-crop → dominant colors → nearest named color (30-color dictionary)
- Write `data/metadata.csv`: filename, image_id, category, color, category_conf, color_conf
- Smoke test: confirm CSV has 3200 rows with valid entries

## Phase 2: Pseudo-Label Generation — Location + Style
**File**: `indexer/generate_pseudo_labels.py`
**Actions**:
- Places365 (ResNet50 via torchvision) → 7 scene categories → pseudo-location + loc_conf
- Vanilla CLIP zero-shot → 6 style prompts → pseudo_style + style_conf
- Append columns to `data/metadata.csv`
- Smoke test: verify all 3200 images labeled

## Phase 3: Embedding + Vector Index
**Files**: `indexer/build_embeddings.py`, `indexer/build_vector_index.py`
**Actions**:
- Fashion CLIP encode → `data/embeddings.npy` (3200×512, L2-normalized)
- FAISS FlatIP index (exact search, given 3200 scale)
- SQLite metadata store: `data/glance.db` with images table

## Phase 4: Query Parser + Search Pipeline
**Files**: `retriever/query_parser.py`, `retriever/search.py`
**Actions**:
- Keyword-based query parser: extract color[], garment[], location?, style?
- Search pipeline: CLIP encode query → FAISS top-50 → reranker (attr_match + cos_sim) → top-k
- Demo: `python -m retriever.search "a blue shirt in a park"`

## Phase 5: Evaluation
**File**: `retriever/evaluate.py`
**Actions**:
- Run all 5 queries against full pipeline
- Record Precision@5, rank of first relevant, failure types
- Ablation: Full → No-color → No-location → No-style → Vanilla CLIP
- Print comparison table

## Phase 6: Manual Validation + Failure Analysis
**Actions**:
- Manual spot-check of 50 random images: category/color/location/style accuracy
- False-positive analysis for each of the 5 queries
- Document findings

## Phase 7: Report
**File**: `docs/report.md`
**Actions**:
- Problem statement, architecture, pipeline details
- Evaluation results table + ablation
- Failure analysis, limitations, scalability discussion
- Deliverable: `docs/report.md`
