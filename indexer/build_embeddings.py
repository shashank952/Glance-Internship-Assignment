"""
Phase 3a — Embedding Generation.

FashionCLIP encode all 3200 images → data/embeddings.npy (3200×512, L2-normalized).
"""

import os, sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from transformers import CLIPModel, CLIPProcessor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE, "data", "sample_images")
META_PATH = os.path.join(BASE, "data", "metadata.csv")
OUT_PATH = os.path.join(BASE, "data", "embeddings.npy")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    df = pd.read_csv(META_PATH)
    print(f"[Embeddings] Loading model on {DEVICE}…")
    model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    model.eval()

    embeddings = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embeddings"):
        img_path = os.path.join(IMAGE_DIR, row["filename"])
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model.vision_model(**inputs)
                embed = model.visual_projection(outputs.pooler_output)
                embed = embed / embed.norm(dim=-1, keepdim=True)
            embeddings.append(embed.cpu().numpy().squeeze(0))
        except Exception as e:
            print(f"\n  [WARN] Failed {row['filename']}: {e}")
            embeddings.append(np.zeros(512, dtype=np.float32))

    emb_array = np.array(embeddings, dtype=np.float32)
    np.save(OUT_PATH, emb_array)
    print(f"[Embeddings] Saved {emb_array.shape} to {OUT_PATH}")


if __name__ == "__main__":
    main()
