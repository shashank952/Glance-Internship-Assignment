"""
Phase 1 — Metadata Generation: Fashion CLIP category + LAB color (full image).

Garment categories expanded to 30 classes including accessories.
Color uses KMeans in CIELAB space on the full image (no center-crop).
"""

import os, sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb

import torch
from transformers import CLIPProcessor, CLIPModel

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE, "data", "sample_images")
OUTPUT_DIR = os.path.join(BASE, "data")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Garment categories (30 classes — expanded with accessories)
# ---------------------------------------------------------------------------
GARMENT_PROMPTS = [
    "dress", "shirt", "t-shirt", "blouse", "tank top",
    "sweater", "cardigan", "jacket", "coat", "blazer",
    "hoodie", "vest",
    "pants", "jeans", "shorts", "skirt",
    "jumpsuit", "romper", "bodysuit", "suit",
    # Accessories
    "tie", "scarf", "hat", "belt",
    "gloves", "sunglasses", "bag", "shoes",
    "boots", "sneakers",
]

# ---------------------------------------------------------------------------
# Named color dictionary — stored in LAB space for perceptual distance
# ---------------------------------------------------------------------------
COLOR_NAMES = [
    # 0-4: primaries
    "red", "green", "blue", "yellow", "white", "black",
    # 6-14: pinks, purples, warm
    "pink", "purple", "orange", "brown", "gray",
    # 11-25: blues and near-blues (expanded capture for blue family — see _color_audit.py)
    "navy", "blue", "blue", "blue", "teal",
    "coral", "lavender", "olive", "mint", "gold",
    "silver", "taupe", "burgundy", "khaki", "indigo",
    # 26-34: remaining
    "turquoise", "magenta", "charcoal", "ivory",
    "beige", "cream", "maroon", "red",
]
COLOR_RGB = np.array([
    # 0-4
    [255,0,0],[0,128,0],[0,0,255],[255,255,0],[255,255,255],[0,0,0],
    # 6-10
    [255,192,203],[128,0,128],[255,165,0],[165,42,42],[128,128,128],
    # 11-20: blues expanded with 3 extra blue variants for coverage
    [0,0,128],        # navy
    [50,80,200],      # medium blue (extra blue prototype)
    [70,130,200],     # steel/denim blue (extra blue prototype)
    [100,150,255],    # light royal blue (extra blue prototype)
    [0,128,128],      # teal
    # 16-20
    [255,127,80],[230,230,250],[128,128,0],[189,252,201],[255,215,0],
    # 21-25
    [192,192,192],[72,60,50],[128,0,32],[195,176,145],[75,0,130],
    # 26-30
    [64,224,208],[255,0,255],[54,69,79],[255,255,240],
    # 31-35 — shifted entries that were bumped
    [245,245,220],[255,253,208],[128,0,0],[200,50,50],  # medium red (extra red prototype)
], dtype=np.float64)

# Pre-compute LAB values for the named color dictionary
COLOR_LAB = rgb2lab(COLOR_RGB[np.newaxis, ...] / 255.0).squeeze(0)


def nearest_named_color(lab_triplet):
    """Map a LAB triplet to the closest named color (LAB Euclidean distance)."""
    distances = np.linalg.norm(COLOR_LAB - lab_triplet.reshape(1, 3), axis=1)
    return COLOR_NAMES[int(np.argmin(distances))]


def extract_dominant_colors(image_path, n_clusters=3):
    """
    Extract dominant colors via KMeans on the FULL image in LAB space.
    Returns list of (color_name, pixel_fraction) sorted descending.
    """
    img = Image.open(image_path).convert("RGB")
    small = img.resize((128, 128))  # downsample for speed
    rgb_pixels = np.array(small).reshape(-1, 3).astype(np.float64) / 255.0
    lab_pixels = rgb2lab(rgb_pixels.reshape(-1, 1, 3)).reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(lab_pixels)
    counts = np.bincount(labels)

    results = []
    for i in range(n_clusters):
        lab_center = kmeans.cluster_centers_[i]
        name = nearest_named_color(lab_center)
        frac = float(counts[i]) / float(len(labels))
        results.append((name, frac))

    results.sort(key=lambda x: -x[1])
    return results


class FashionCLIPClassifier:
    """Zero-shot garment classifier using patrickjohncyh/fashion-clip."""

    def __init__(self):
        print(f"[FashionCLIP] Loading model on {DEVICE}…")
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        texts = [f"a photo of a {p}" for p in GARMENT_PROMPTS]
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            text_outputs = self.model.text_model(**inputs)
            self.text_embeds = self.model.text_projection(text_outputs.pooler_output)
            self.text_embeds = self.text_embeds / self.text_embeds.norm(dim=-1, keepdim=True)

    def classify(self, image_path):
        """Return (category_name, confidence) for a single image."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            vision_outputs = self.model.vision_model(**inputs)
            image_embeds = self.model.visual_projection(vision_outputs.pooler_output)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        similarities = (image_embeds @ self.text_embeds.T).squeeze(0).cpu().numpy()
        exp_s = np.exp(similarities - similarities.max())
        probs = exp_s / exp_s.sum()
        best_idx = int(np.argmax(similarities))
        return GARMENT_PROMPTS[best_idx], float(probs[best_idx])


def main():
    classifier = FashionCLIPClassifier()

    image_paths = sorted(
        [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )
    print(f"[Phase 1] Found {len(image_paths)} images.")

    rows = []
    for idx, img_path in enumerate(tqdm(image_paths, desc="Category + Color")):
        filename = os.path.basename(img_path)

        try:
            category, category_conf = classifier.classify(img_path)
        except Exception as e:
            print(f"\n  [WARN] Category failed for {filename}: {e}")
            category, category_conf = "unknown", 0.0

        try:
            colors = extract_dominant_colors(img_path, n_clusters=3)
            c1, c1c = colors[0][0], round(colors[0][1], 4)
            c2 = colors[1][0] if len(colors) > 1 else ""
            c2c = round(colors[1][1], 4) if len(colors) > 1 else 0.0
            c3 = colors[2][0] if len(colors) > 2 else ""
            c3c = round(colors[2][1], 4) if len(colors) > 2 else 0.0
        except Exception as e:
            print(f"\n  [WARN] Color failed for {filename}: {e}")
            c1, c1c = "unknown", 0.0
            c2, c2c = "", 0.0
            c3, c3c = "", 0.0

        rows.append({
            "filename": filename,
            "image_id": idx,
            "category": category,
            "color": c1,
            "color_conf": c1c,
            "color2": c2,
            "color2_conf": c2c,
            "color3": c3,
            "color3_conf": c3c,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[Phase 1] Wrote {len(df)} rows to {OUTPUT_CSV}")
    print(df[["filename", "category", "color", "color_conf", "color2", "color2_conf", "color3", "color3_conf"]].head(10).to_string())


if __name__ == "__main__":
    main()
