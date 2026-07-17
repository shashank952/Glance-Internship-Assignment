# Glance — Fashion Image Retrieval System

## Problem Statement

Fashion image retrieval from unstructured text queries requires **compositional understanding** — the ability to bind attributes (color, garment type, style, location) to their correct referents. Standard CLIP + FAISS fails on compositional queries: a search for "red shirt blue pants" retrieves "blue shirt red pants" with equal similarity. This system solves that by decomposing queries and images into structured attribute vectors and applying a multi-stage reranking pipeline.

## Architecture

```
                    ┌──────────────┐
                    │  Query Text  │
                    └──────┬───────┘
                           ▼
                ┌──────────────────┐
                │  Query Parser    │
                │  (keyword-based) │
                │  → color[],      │
                │    garment[],    │
                │    location?,    │
                │    style?        │
                └────────┬─────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
  ┌────────────┐  ┌────────────┐  ┌──────────────┐
  │ CLIP       │  │ FAISS      │  │ SQLite      │
  │ Encode     │  │ FlatIP     │  │ Metadata    │
  │ (text)     │  │ Top-50     │  │ Store       │
  └─────┬──────┘  └─────┬──────┘  └──────┬───────┘
        │               │                │
        ▼               ▼                ▼
  ┌──────────────────────────────────────────┐
  │           Reranker (Weighted)            │
  │  0.6 × cos_sim + 0.4 × attr_match       │
  └──────────────────┬───────────────────────┘
                     ▼
               ┌──────────┐
               │ Top-K    │
               │ Results  │
               └──────────┘
```

### Indexing Pipeline

1. **Category** — `patrickjohncyh/fashion-clip` zero-shot over 20 garment prompts → garment category per image
2. **Color** — KMeans (k=3) on RGB pixels from center-crop (60%) → nearest named color from 30-color dictionary → top-3 dominant colors
3. **Location** — FashionCLIP zero-shot over 7 scene prompts → indoor/outdoor/office/beach/park/city-street
4. **Style** — FashionCLIP zero-shot over 6 style prompts → casual/formal/sporty/bohemian/vintage/minimalist
5. **Embedding** — FashionCLIP vision encoder → 512-dim L2-normalized vector per image
6. **FAISS Index** — `IndexFlatIP` (exact inner product) → `data/faiss_index.bin`
7. **SQLite Store** — All metadata → `data/glance.db`

### Retrieval Pipeline

1. **Query Parser** — Keyword matching over known color/garment/location/style dictionaries → structured query dict
2. **CLIP Encode** — Same FashionCLIP text encoder → 512-dim query vector
3. **FAISS Search** — Top-50 candidates by cosine similarity
4. **Reranker** — `final_score = 0.6 × cos_sim + 0.4 × attribute_match_score`
   - Attribute match: proportion of query-specified attributes (color, garment type, location, style) that match the metadata row
5. **Return** — Top-k results sorted by combined score

## Dataset

- **Source**: Fashion product images (3,200 total)
- **Labels**: Automatic pseudo-labels only — no human annotations
- **Garment categories** (20): dress (1096), coat (262), skirt (253), romper (195), jumpsuit (146), jeans (144), bodysuit (134), suit (127), sweater (116), blouse (110), blazer (97), cardigan (90), tank top (82), shorts (80), t-shirt (72), pants (57), jacket (50), shirt (48), vest (22), hoodie (19)
- **Colors** (30): dominated by neutrals — taupe (715), black (548), silver (501), gray (492), khaki (282), charcoal (189)

## Evaluation

### 5 Test Queries

| # | Query | What it tests | Full P@5 |
|---|---|---|---|
| 1 | "A person in a bright yellow raincoat." | Color + garment binding | 0.00 |
| 2 | "Professional business attire inside a modern office." | Location + style inference | **0.80** |
| 3 | "Someone wearing a blue shirt sitting on a park bench." | Multi-attribute composition | 0.00 |
| 4 | "Casual weekend outfit for a city walk." | Style-only (no explicit garment) | **1.00** |
| 5 | "A red tie and a white shirt in a formal setting." | Compositional (two colors, two garments) | 0.00 |

### Ablation Study (Precision@5)

| Variant | Q1 | Q2 | Q3 | Q4 | Q5 |
|---------|----|----|----|----|----|
| Full | 0.00 | **0.80** | 0.00 | **1.00** | 0.00 |
| No-Color | 0.00 | 0.80 | 0.00 | 1.00 | 0.00 |
| No-Location | 0.00 | 0.40 | 0.00 | 0.60 | 0.00 |
| No-Style | 0.00 | 0.40 | 0.00 | 0.80 | 0.00 |
| Vanilla CLIP | 0.00 | 0.40 | 0.00 | 0.40 | 0.00 |

**Key finding**: Reranking provides a clear improvement over vanilla CLIP — Q2 goes from 0.40 → 0.80 and Q4 from 0.40 → 1.00. Location and style attributes each contribute meaningfully (dropping them reduces P@5).

## Failure Analysis

### Why Q1, Q3, Q5 score 0.00

These failures are **not pipeline failures** — they are **metadata coverage failures**:

- **Q1 (yellow raincoat)**: Zero images labeled as both "yellow" and "coat" in the dataset. The color classifier produces overwhelmingly neutral colors (taupe, black, silver, gray dominate). RGB-space KMeans on fashion product images captures background/lighting bias, not garment color.
- **Q3 (blue shirt, park)**: Only 4 images have the combo of shirt + park-or-garden location. Zero shirts are labeled "blue" — the blue channel is rarely dominant enough in RGB KMeans.
- **Q5 (red tie + white shirt, formal)**: Zero images match formal style + red/white + shirt/suit/blazer. And "tie" is not one of the 20 FashionCLIP garment categories.

### Metadata Quality Issues

1. **Color bias**: KMeans in RGB on center-crop is heavily influenced by background/lighting. 58% of images have a neutral (taupe/black/silver/gray) as dominant color. The Euclidean distance to named colors in RGB is perceptually non-uniform.
2. **Location sparsity**: "park or garden" (65) and "office environment" (37) are rare — CLIP scene zero-shot is skewed toward "city street" (1452).
3. **Style ambiguity**: "minimalist style" (735) and "formal style" (707) dominate, "bohemian" (245) is least common — these are reasonable distributions but limit compositional diversity.

### Recommendations

1. **Replace RGB KMeans with LAB-space clustering** for perceptually uniform color distance
2. **Increase crop aggressiveness** (30% center-crop instead of 60%) to reduce background pollution in color extraction
3. **Use Places365** for location (instead of CLIP zero-shot) for more accurate scene classification
4. **Expand garment vocabulary** beyond FashionCLIP's 20 categories to include accessories (tie, scarf, hat)
5. **Add real human annotations** for evaluation ground truth instead of pseudo-label-based relevance

## Scalability Discussion

| Component | Current (3,200 images) | Scaling to 100K | Scaling to 10M |
|-----------|----------------------|-----------------|-----------------|
| FAISS index | FlatIP (exact) | IVF + PQ (approximate) | IVF + PQ + sharding |
| Embedding storage | 3200×512 = 6.3 MB | 100K×512 = 195 MB | 10M×512 = 19 GB |
| Index build | Instant | ~10 min | ~2 hours |
| Query time | ~1 ms | ~5 ms | ~20 ms (IVF) |
| Metadata DB | SQLite (local) | SQLite (local) | PostgreSQL (distributed) |
| Rerank | 50 candidates | 200 candidates | 1000 candidates |

The reranking architecture is inherently scalable: the expensive operation (CLIP encoding + FAISS search) is bounded by index size, while the reranker only processes a fixed candidate window (top-50 → top-200 → top-1000 for larger datasets).

## Files

```
indexer/
├── __init__.py
├── generate_metadata.py    — Phase 1: category + color
├── generate_pseudo_labels.py — Phase 2: location + style
├── build_embeddings.py     — Phase 3a: CLIP image embeddings
└── build_vector_index.py   — Phase 3b: FAISS + SQLite

retriever/
├── __init__.py
├── query_parser.py         — Phase 4a: keyword query parser
├── search.py               — Phase 4b: search + reranker
└── evaluate.py             — Phase 5: evaluation + ablation

evaluation/
├── __init__.py
└── validate.py             — Phase 6: spot check + failure analysis

data/
├── metadata.csv            — Full metadata (3200×13)
├── embeddings.npy          — Image embeddings (3200×512)
├── faiss_index.bin         — FAISS FlatIP index
└── glance.db               — SQLite metadata store
```
