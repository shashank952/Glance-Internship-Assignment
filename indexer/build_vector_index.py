"""
Phase 3b — Vector Index + SQLite Metadata Store.

Reads data/embeddings.npy + data/metadata.csv → builds:
  - data/faiss_index.bin (FAISS FlatIP)
  - data/glance.db (SQLite metadata)
"""

import os, sys
import sqlite3
import numpy as np
import pandas as pd
import faiss

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_PATH = os.path.join(BASE, "data", "embeddings.npy")
META_PATH = os.path.join(BASE, "data", "metadata.csv")
INDEX_PATH = os.path.join(BASE, "data", "faiss_index.bin")
DB_PATH = os.path.join(BASE, "data", "glance.db")


def build_faiss(embeddings):
    """Build FlatIP index (exact search, appropriate for 3K scale)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    print(f"[FAISS] Index with {index.ntotal} vectors written to {INDEX_PATH}")
    return index


def build_sqlite(df):
    """Populate SQLite metadata store."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS images (
            image_id    INTEGER PRIMARY KEY,
            filename    TEXT,
            category    TEXT,
            color       TEXT,
            color_conf  REAL,
            color2      TEXT,
            color2_conf REAL,
            color3      TEXT,
            color3_conf REAL,
            location    TEXT,
            loc_conf    REAL,
            style       TEXT,
            style_conf  REAL
        )
    """)
    cur.execute("DELETE FROM images")
    df.to_sql("images", conn, if_exists="append", index=False)
    conn.commit()
    row_count = cur.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    conn.close()
    print(f"[SQLite] {row_count} rows written to {DB_PATH}")


def main():
    embeddings = np.load(EMB_PATH)
    print(f"[VectorIndex] Loaded embeddings: {embeddings.shape}")

    # Verify L2-normalized
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"[VectorIndex] Norm range: {norms.min():.4f} – {norms.max():.4f}")

    build_faiss(embeddings)
    df = pd.read_csv(META_PATH)
    build_sqlite(df)
    print("[VectorIndex] Done.")


if __name__ == "__main__":
    main()
