# Product Requirements Document (PRD)
## Multimodal Fashion & Context Retrieval System
**Project:** Glance ML Internship Assignment
**Owner:** Shashank
**Version:** 1.0
**Date:** July 2026

---

## 1. Problem Statement

Build a system that retrieves fashion images from a database using free-text natural language queries. The system must jointly understand three things at once, not in isolation:

- **What** — clothing type and color (e.g., "yellow raincoat", "blue shirt")
- **Where** — the environment/context (office, street, park, home)
- **Vibe** — the style (formal, casual, weekend, professional)

The core technical challenge called out by the assignment: **vanilla CLIP is not enough**. CLIP's contrastive image-text embedding is a strong zero-shot baseline but is known to behave like a "bag of concepts" — it often fails at **compositionality** (e.g., "red shirt with blue pants" vs "blue shirt with red pants" can score similarly). The system must be architected to overcome this specific weakness.

---

## 2. Goals & Non-Goals

### Goals
- Retrieve top-k relevant images for natural language queries covering attribute, location, style, and compositional prompts.
- Demonstrate an approach that is measurably better than naive CLIP embedding + cosine similarity, specifically on compositional queries.
- Keep code modular: indexing logic, retrieval logic, and data are cleanly separated.
- Design the retrieval logic so it is credible at ~1M image scale (even though the working dataset is 3200 images).
- Handle **zero-shot** queries — phrasing not seen during any labeling/training step.

### Non-Goals
- Building a custom vector database from scratch (use FAISS — off-the-shelf).
- Training a large model from scratch (fine-tuning small adapters/classifiers is fine; full pretraining is out of scope).
- Perfect location detection — location tags are a supporting signal, not a ground-truth requirement (no dataset provides this natively).
- Production deployment (auth, UI polish, hosting) — this is an ML/retrieval quality exercise, not a full product.

---

## 3. Dataset

### 3.1 Source
**Provided dataset — 3200 fashion images. No metadata provided.**

Images have hash-based filenames (32 hex characters) and come without any labels. All attributes — category, color, location, style — must be generated from pixel content alone.

Place images in `data/raw_images/`. The indexer generates `data/metadata.csv` by extracting category (Fashion CLIP), color (KMeans), pseudo-location (Places365), and pseudo-style (vanilla CLIP). See Section 5.2 for the full pseudo-labeling pipeline.

### 3.2 No sampling needed
All 3200 images are used. No downsampling required.

### 3.3 File naming convention
Use a neutral, non-descriptive filename tied to the provided image ID — never encode the answer (color/location/style) into the filename itself. This avoids any temptation for the retrieval system to "cheat" via filename matching, and keeps naming consistent with the assignment's explicit "no filename/keyword matching" requirement.

```
img_00001.jpg
img_00002.jpg
```

All actual attributes (category, color, pseudo-location, pseudo-style) live in a separate metadata table (CSV/JSON), keyed by filename/image_id — not in the filename.

| filename | image_id | category | color | pseudo_location | pseudo_style |
|---|---|---|---|---|---|
| img_00001.jpg | 1 | jacket | blue | street | casual |
| img_00002.jpg | 2 | dress | red | office | formal |

---

## 4. System Architecture Overview

```
                 ┌─────────────────────────┐
                 │   Provided Dataset      │
                 │   (3200 fashion images) │
                 └───────────┬─────────────┘
                            │
                 ┌──────────▼───────────┐
                 │  Metadata Enrichment  │
                 │  (Part of Indexer)    │
                 │  - garment category   │
                 │  - color attribute    │
                 │  - pseudo-location    │
                 │  - pseudo-style       │
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │   Embedding Layer     │
                 │  CLIP/SigLIP image    │
                 │  encoder + fashion-   │
                 │  aware attribute      │
                 │  vector (fused)       │
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │   Vector Database     │
                 │  (FAISS)              │
                 └──────────┬───────────┘
                            │
   query ─▶ ┌──────────────▼───────────────┐
            │        RETRIEVER              │
            │ 1. Parse query → structured   │
            │    attributes (color/type/    │
            │    location/style) via LLM    │
            │    or rule-based extractor    │
            │ 2. Embed query text (CLIP)    │
            │ 3. Vector similarity search    │
            │    (top ~50 candidates)        │
            │ 4. Re-rank candidates using    │
            │    structured attribute match  │
            │    (color, category, location) │
            │ 5. Return top-k final results  │
            └───────────────┬────────────────┘
                            │
                     Top-k image results
```

---

## 5. Technical Requirements

### 5.1 Part A — The Indexer

**Responsibilities:**
1. Load raw images + provided metadata.
2. Generate pseudo-location and pseudo-style tags (see 5.2).
3. Compute image embeddings using a pretrained vision-language encoder (CLIP or SigLIP).
4. Compute/store a **structured attribute vector** per image (category, color, location, style) — this is the fashion-aware layer that goes beyond vanilla CLIP.
5. Store both the dense embedding and the structured metadata in **FAISS** (IVF-PQ or HNSW index for scalability). Metadata filtering is handled by a separate lightweight index (e.g., a dict or SQLite) keyed on image_id — FAISS returns candidate IDs, then the metadata filter + reranker runs on the shortlist.

**Key design decision:** Do not rely on raw CLIP embedding similarity alone. Fuse:
- Dense embedding (semantic/visual similarity)
- Structured attribute match score (precise color/category/location match)

This directly solves the compositionality problem called out in the assignment hint — "red shirt with blue pants" vs "blue shirt with red pants" get correctly disambiguated because color is tied to a structured field, not just blended into one embedding vector.

### 5.2 Pseudo-Label Generation (the "beyond CLIP" component)

Since no dataset has location or style labels, generate them programmatically. Category is also generated since the provided images come with zero metadata.

#### Category (Fashion CLIP)

Use **Fashion CLIP** (`patrickjohncyh/fashion-clip`) — fine-tuned on 800K fashion captions. Classify each image against 20 garment prompts (dress, shirt, jacket, pants, etc.) and take the highest-scoring match. Fashion CLIP knows fashion terminology that vanilla CLIP collapses (blazer vs. suit jacket, crop top vs. tank top).

#### Color (KMeans in LAB space)

Unsupervised pixel-level color extraction. Center-crop each image (garment is usually centered), convert RGB→LAB (perceptually uniform), run KMeans(k=3), map dominant cluster to nearest named color. More reliable than CLIP for fine-grained color.

#### Location (Places365)

Use a **Places365** pretrained scene classifier (ResNet50) instead of CLIP zero-shot. Places365 was trained on 365 scene categories with 10M+ labeled images — it knows what an office, street, or park looks like with ~70–80% accuracy vs. CLIP's ~40–50% for zero-shot scene classification.

Relevant scene categories: `office_general`, `street`, `park`, `living_room`, `dining_room`, `beach`, `studio`.

#### Style (vanilla CLIP zero-shot)

No dedicated fashion-style classifier exists. Classify against prompts:
`["casual / everyday wear", "formal / business attire", "athletic / sportswear", "evening / night out", "summer / vacation outfit", "winter / cold weather outfit"]`

Take the highest-scoring prompt as pseudo-label. This is the weakest label — confidence below 0.3 should be discarded in the reranker.

#### Confidence scores

Store confidence for every label (category_conf, color_conf, loc_conf, style_conf). The reranker uses these to dynamically downweight unreliable labels at query time. Images where any confidence is below 0.3 are candidates for manual spot-checking in the failure analysis.

### 5.3 Part B — The Retriever

**Responsibilities:**
1. Accept a natural language query string.
2. **Query understanding** — extract structured attributes from the query (color, garment type, location, style) using either:
   - A lightweight rule-based/keyword+synonym extractor, or
   - An LLM-based structured extraction call (e.g., prompt a small LLM to output JSON: `{color: "blue", garment: "shirt", location: "park", style: null}`)
3. Embed the raw query text with the same CLIP/SigLIP encoder used for images.
4. Run vector similarity search against the indexed embeddings to get a candidate shortlist (e.g., top 50).
5. **Re-rank** that shortlist using the structured attribute match (does the image's color tag match the query's extracted color? Does its pseudo-location match?). This is the step that fixes compositional queries.
6. Return the final top-k images, ranked by combined score (embedding similarity + attribute match bonus).

**Why this two-stage design (retrieve-then-rerank) matters for scalability:** At 1M images, doing expensive per-item attribute comparison against every image is wasteful. Vector search narrows 1M → ~50 candidates cheaply (this is what FAISS is built for), and the more expensive structured re-ranking only runs on that small shortlist. This pattern is standard in production search systems and is explicitly what makes the design credible at scale.

### 5.4 Evaluation Queries (from assignment) — how each maps to the pipeline

| # | Query | What it stresses | How the design handles it |
|---|---|---|---|
| 1 | "A person in a bright yellow raincoat." | Attribute (color + garment) | Structured color+category match dominates re-rank |
| 2 | "Professional business attire inside a modern office." | Location + style | Pseudo-location tag + pseudo-style tag match |
| 3 | "Someone wearing a blue shirt sitting on a park bench." | Complex/multi-attribute | Embedding similarity (pose/scene) + color/location re-rank |
| 4 | "Casual weekend outfit for a city walk." | Style inference, no explicit garment | Style pseudo-label + embedding similarity carries this (no garment to anchor on — good zero-shot test) |
| 5 | "A red tie and a white shirt in a formal setting." | Compositional (two colors, two garments) | This is the key test for the attribute-fusion layer vs vanilla CLIP — structured per-garment color match prevents red/white swap confusion |

---

## 6. Tech Stack

| Component | Choice | Why |
|---|---|---|
| Image/text encoder | Fashion CLIP (patrickjohncyh/fashion-clip) | Fine-tuned on 800K fashion captions — better fashion concept alignment than vanilla CLIP |
| Scene classifier | Places365 (ResNet50) | Dedicated scene model beats CLIP zero-shot by ~30 points for location tagging |
| Color extraction | KMeans in LAB space | Unsupervised, more reliable than CLIP for fine color discrimination |
| Vector DB | FAISS (IVF-PQ / HNSW) | Industry standard, scales to 1M+; metadata filter in companion SQLite |
| Query parsing | Small LLM (via API) or rule-based extractor | Converts free text into structured attributes |
| Data handling | pandas (provided CSV/JSON) | Simple loading + metadata management |
| Language | Python | Ecosystem fit for all of the above |

---

## 7. Repository Structure

```
fashion-context-retrieval/
├── README.md
├── requirements.txt
├── data/
│   ├── raw_images/                # provided 3200 images
│   └── metadata.csv               # category, color, pseudo_location, pseudo_style
├── indexer/
│   ├── generate_pseudo_labels.py  # zero-shot location/style tagging
│   ├── build_embeddings.py        # CLIP/SigLIP encoding
│   └── build_vector_index.py      # build FAISS index + metadata store
├── retriever/
│   ├── query_parser.py            # extract structured attributes from text
│   ├── search.py                  # FAISS search + re-rank logic
│   └── evaluate.py                # run the 5 evaluation queries + inspect results
├── notebooks/
│   └── exploration.ipynb
└── docs/
    └── PRD.md                     # this document
```

Logic and data are fully separated — swapping the dataset or the encoder model requires no changes to `retriever/`.

---

## 8. Future Improvements (per assignment ask)

**Extending location/weather:**
- Replace zero-shot pseudo-labels with a fine-tuned scene classifier trained on a small manually-labeled seed set (few hundred images) for higher precision.
- Add weather as a separate zero-shot axis (`"sunny", "rainy", "snowy", "indoor"`), same pattern as location.
- Incorporate EXIF/geo metadata if working with a real photo corpus instead of a curated dataset.

**Improving precision:**
- Fine-tune Fashion CLIP's text/image projection heads on fashion-specific attribute pairs (contrastive fine-tuning) rather than relying purely on zero-shot.
- Add a learned re-ranker (small cross-encoder) instead of hand-weighted attribute-match scoring.
- Add object detection (Grounding DINO) for per-garment color extraction — this is the missing piece that would solve query 5 binding fully.
- Expand the structured attribute schema (fabric, pattern, fit) using the dataset's full attribute ontology instead of only category+color.
- Add hard-negative mining during evaluation to specifically stress-test compositional confusions (the red-shirt/blue-pants case).

---

## 9. Risks / Open Questions

- Pseudo-labels for location (Places365) and style (CLIP) are noisy by construction — Places365 is more reliable (~70–80%) but style labels remain weak. Worth manually spot-checking a sample (e.g., 50 images) to report per-attribute accuracy in the write-up.
- Fashion CLIP improves category accuracy over vanilla CLIP but is still zero-shot — it can fail on unusual garment types or ambiguous angles. The confidence score catch this.
- The critical unsolved problem is per-garment color binding without object detection. The reranker improves results but doesn't fully solve query 5's attribute-object swap. The report should acknowledge this explicitly and propose Grounding DINO as the fix.
- FAISS is already chosen for scalability. The IVF-PQ or HNSW index parameters must be tuned to the 3200-image working set — small enough that brute-force exact search may be competitive, but the index structure should demonstrate the production-ready approach. Worth reporting exact vs. approximate recall tradeoffs in the write-up.