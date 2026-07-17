"""
Phase 5 — Evaluation + Ablation Study.

Runs all 5 queries against the full pipeline and ablations.
Reports Precision@5, rank of first relevant, and failure analysis.
"""

import os, sys
import numpy as np
import pandas as pd
import sqlite3
import faiss
from copy import deepcopy

import torch
from transformers import CLIPModel, CLIPProcessor

from retriever.query_parser import parse_query

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE, "data", "faiss_index.bin")
DB_PATH = os.path.join(BASE, "data", "glance.db")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Evaluation queries (from AGENTS.md / PRD)
# ---------------------------------------------------------------------------
EVAL_QUERIES = [
    {
        "id": 1,
        "query": "A person in a bright yellow raincoat.",
        "criteria": lambda r: r.get("color") == "yellow" and r.get("category") in ("coat",),
    },
    {
        "id": 2,
        "query": "Professional business attire inside a modern office.",
        "criteria": lambda r: r.get("location") in ("indoor setting", "office environment")
                              and r.get("style") == "formal style",
    },
    {
        "id": 3,
        "query": "Someone wearing a blue shirt sitting on a park bench.",
        "criteria": lambda r: r.get("color") == "blue"
                              and r.get("category") in ("shirt", "t-shirt", "blouse")
                              and r.get("location") == "park or garden",
    },
    {
        "id": 4,
        "query": "Casual weekend outfit for a city walk.",
        "criteria": lambda r: r.get("style") == "casual style"
                              and r.get("location") == "city street",
    },
    {
        "id": 5,
        "query": "A red tie and a white shirt in a formal setting.",
        "criteria": lambda r: (r.get("color") in ("red", "white") or r.get("color2") in ("red", "white"))
                              and r.get("category") in ("shirt", "suit", "blazer", "tie")
                              and r.get("style") == "formal style",
    },
]


class EvalSearch:
    """Search with configurable rerank weights (for ablation)."""

    def __init__(self):
        self.index = faiss.read_index(INDEX_PATH)
        self.conn = sqlite3.connect(DB_PATH)
        cur = self.conn.execute("SELECT * FROM images ORDER BY image_id")
        cols = [d[0] for d in cur.description]
        self.rows = [dict(zip(cols, r)) for r in cur.fetchall()]

        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        self.model.eval()

    def encode_query(self, query: str) -> np.ndarray:
        inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = self.model.text_model(**inputs)
            embed = self.model.text_projection(outputs.pooler_output)
            embed = embed / embed.norm(dim=-1, keepdim=True)
        return embed.cpu().numpy().squeeze(0).astype(np.float32)

    def attr_score(self, row: dict, parsed: dict, weights: dict = None) -> float:
        """Attribute match score with per-component weights for ablation."""
        if weights is None:
            weights = {"color": 1.0, "garment": 1.0, "location": 1.0, "style": 1.0}

        score = 0.0
        total_weight = 0.0

        if parsed["colors"] and weights["color"] > 0:
            db_colors = {row.get("color", ""), row.get("color2", ""), row.get("color3", "")}
            db_colors.discard("")
            if any(c in db_colors for c in parsed["colors"]):
                score += weights["color"]
            total_weight += weights["color"]

        if parsed["garments"] and weights["garment"] > 0:
            db_garment = row.get("category", "")
            if any(g in db_garment or db_garment.startswith(g) or g.startswith(db_garment)
                   for g in parsed["garments"]):
                score += weights["garment"]
            total_weight += weights["garment"]

        if parsed["location"] and weights["location"] > 0:
            if row.get("location", "") == parsed["location"]:
                score += weights["location"]
            total_weight += weights["location"]

        if parsed["style"] and weights["style"] > 0:
            if row.get("style", "") == parsed["style"]:
                score += weights["style"]
            total_weight += weights["style"]

        return score / max(total_weight, 1.0)

    def search(self, query: str, top_k: int = 10,
               weights: dict = None, vanilla: bool = False) -> list:
        """Search with rerank weights (None = default all 1.0). vanilla=True skips rerank."""
        parsed = parse_query(query)
        q_vec = self.encode_query(query).reshape(1, -1)

        n_candidates = min(50, self.index.ntotal)
        scores_50, indices_50 = self.index.search(q_vec, n_candidates)

        if vanilla:
            # Pure cosine similarity, no rerank
            results = []
            for score, idx in zip(scores_50[0], indices_50[0]):
                row = self.rows[idx]
                results.append({**row, "cosine_score": float(score),
                                "attr_score": 0.0, "combined_score": float(score)})
            return results[:top_k]

        candidates = []
        for score, idx in zip(scores_50[0], indices_50[0]):
            row = self.rows[idx]
            a_score = self.attr_score(row, parsed, weights)
            combined = 0.6 * float(score) + 0.4 * a_score
            candidates.append({**row, "cosine_score": float(score),
                               "attr_score": a_score,
                               "combined_score": combined})

        candidates.sort(key=lambda x: x["combined_score"], reverse=True)
        return candidates[:top_k]


def evaluate_query(search, qdef, weights=None, vanilla=False):
    """Run a query and compute metrics. Returns dict."""
    top5 = search.search(qdef["query"], top_k=5, weights=weights, vanilla=vanilla)
    first_rel = None
    rel_count = 0
    for rank, r in enumerate(top5, 1):
        if qdef["criteria"](r):
            rel_count += 1
            if first_rel is None:
                first_rel = rank

    precision_at_5 = rel_count / 5.0
    return {
        "query_id": qdef["id"],
        "query": qdef["query"],
        "precision@5": precision_at_5,
        "first_relevant_rank": first_rel if first_rel else "—",
    }


def main():
    search = EvalSearch()

    # ---- Ablation configurations ----
    variants = [
        ("Full", None, False),
        ("No-Color", {"color": 0.0, "garment": 1.0, "location": 1.0, "style": 1.0}, False),
        ("No-Location", {"color": 1.0, "garment": 1.0, "location": 0.0, "style": 1.0}, False),
        ("No-Style", {"color": 1.0, "garment": 1.0, "location": 1.0, "style": 0.0}, False),
        ("Vanilla CLIP", None, True),
    ]

    # ---- Run all variants ----
    variant_results = []
    for var_name, weights, vanilla in variants:
        row = {"Variant": var_name}
        for qdef in EVAL_QUERIES:
            res = evaluate_query(search, qdef, weights=weights, vanilla=vanilla)
            row[f"Q{qdef['id']} P@5"] = f"{res['precision@5']:.2f}"
            row[f"Q{qdef['id']} R@1"] = str(res["first_relevant_rank"])
        variant_results.append(row)

    # ---- Summary table ----
    df = pd.DataFrame(variant_results)
    print("\n" + "=" * 90)
    print("EVALUATION — Precision@5 & First Relevant Rank (R@1)")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)
    print()

    # ---- Per-query failure analysis (Full pipeline) ----
    print("—" * 90)
    print("FAILURE ANALYSIS — Full pipeline")
    print("—" * 90)
    for qdef in EVAL_QUERIES:
        top10 = search.search(qdef["query"], top_k=10, weights=None, vanilla=False)
        parsed = parse_query(qdef["query"])
        rel_count_10 = sum(1 for r in top10 if qdef["criteria"](r))

        # Find first relevant
        first_r = None
        for rank, r in enumerate(top10, 1):
            if qdef["criteria"](r):
                first_r = rank
                break

        # Collect top-3 attributes
        top_attrs = []
        for r in top10[:3]:
            top_attrs.append(f"{r['filename'][:24]:24s} cat={r['category']:10s} col={r['color']:10s} "
                           f"loc={str(r.get('location',''))[:14]:14s} sty={str(r.get('style',''))[:14]:14s}")

        print(f"\nQ{qdef['id']}: {qdef['query']}")
        print(f"  Parsed: {parsed}")
        print(f"  Relevant in top-10: {rel_count_10}")
        print(f"  First relevant at rank: {first_r if first_r else '—'}")
        print(f"  Top-3:")
        for line in top_attrs[:3]:
            print(f"    {line}")
    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
