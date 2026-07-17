"""
Search Pipeline — Retriever + Reranker.

Flow:
  1. Parse query → structured attributes
  2. CLIP encode query text → FAISS top-50 (cosine similarity)
  3. Reranker: combine cos_sim + attribute match score
  4. Return top-k results
"""

import os, sys
import sqlite3
import numpy as np
import faiss

import torch
from transformers import CLIPModel, CLIPProcessor

from retriever.query_parser import parse_query, KNOWN_COLORS, KNOWN_GARMENTS, KNOWN_LOCATIONS, KNOWN_STYLES

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE, "data", "faiss_index.bin")
DB_PATH = os.path.join(BASE, "data", "glance.db")
META_PATH = os.path.join(BASE, "data", "metadata.csv")
IMAGE_DIR = os.path.join(BASE, "DATASET")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GlanceSearch:
    """End-to-end search: parse → embed → FAISS → rerank."""

    def __init__(self):
        print(f"[Search] Loading index…")
        self.index = faiss.read_index(INDEX_PATH)
        self.conn = sqlite3.connect(DB_PATH)
        # Load all metadata into a list for fast reranking
        cur = self.conn.execute("SELECT * FROM images ORDER BY image_id")
        cols = [d[0] for d in cur.description]
        self.rows = [dict(zip(cols, r)) for r in cur.fetchall()]

        print(f"[Search] Loading CLIP model on {DEVICE}…")
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        self.model.eval()

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query text → L2-normalized embedding."""
        inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = self.model.text_model(**inputs)
            embed = self.model.text_projection(outputs.pooler_output)
            embed = embed / embed.norm(dim=-1, keepdim=True)
        return embed.cpu().numpy().squeeze(0).astype(np.float32)

    def attribute_match_score(self, row: dict, parsed: dict) -> float:
        """Compute attribute match score (0–1) between parsed query and metadata row."""
        hits = 0
        total = 0

        # Color match
        if parsed["colors"]:
            total += 1
            db_colors = {row.get("color", ""), row.get("color2", ""), row.get("color3", "")}
            db_colors.discard("")
            if any(c in db_colors for c in parsed["colors"]):
                hits += 1

        # Garment match (category column)
        if parsed["garments"]:
            total += 1
            db_garment = row.get("category", "")
            # Partial match: if query has "shirt" and DB has "t-shirt", still counts
            if any(g in db_garment or db_garment in g for g in parsed["garments"]):
                hits += 1

        # Location match
        if parsed["location"]:
            total += 1
            if row.get("location", "") == parsed["location"]:
                hits += 1

        # Style match
        if parsed["style"]:
            total += 1
            if row.get("style", "") == parsed["style"]:
                hits += 1

        return hits / max(total, 1)

    def search(self, query: str, top_k: int = 10) -> list:
        """Run full search pipeline. Returns list of dicts with metadata + scores."""
        parsed = parse_query(query)

        # 1. Encode query
        q_vec = self.encode_query(query).reshape(1, -1)

        # 2. FAISS top-50
        n_candidates = min(50, self.index.ntotal)
        scores_50, indices_50 = self.index.search(q_vec, n_candidates)

        # 3. Rerank
        candidates = []
        for score, idx in zip(scores_50[0], indices_50[0]):
            row = self.rows[idx]
            attr_score = self.attribute_match_score(row, parsed)
            # Combined: 60% cosine similarity, 40% attribute match
            combined = 0.6 * float(score) + 0.4 * attr_score
            candidates.append({
                "image_id": row["image_id"],
                "filename": row["filename"],
                "category": row["category"],
                "color": row["color"],
                "location": row["location"],
                "style": row["style"],
                "cosine_score": round(float(score), 4),
                "attr_score": round(attr_score, 4),
                "combined_score": round(combined, 4),
            })

        # Sort by combined score descending
        candidates.sort(key=lambda x: x["combined_score"], reverse=True)
        return candidates[:top_k]


def demo():
    search = GlanceSearch()
    queries = [
        "A person in a bright yellow raincoat.",
        "Professional business attire inside a modern office.",
        "Someone wearing a blue shirt sitting on a park bench.",
        "Casual weekend outfit for a city walk.",
        "A red tie and a white shirt in a formal setting.",
    ]
    for q in queries:
        parsed = parse_query(q)
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print(f"Parsed: {parsed}")
        results = search.search(q, top_k=5)
        print(f"{'Rank':>4} {'File':<40} {'Category':<12} {'Color':<10} {'CosSim':>7} {'Attr':>5} {'Comb':>5}")
        print("-" * 90)
        for rank, r in enumerate(results, 1):
            print(f"{rank:>4} {r['filename']:<40} {r['category']:<12} {r['color']:<10} {r['cosine_score']:>7.3f} {r['attr_score']:>5.2f} {r['combined_score']:>5.3f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    demo()
