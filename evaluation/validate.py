"""
Phase 6 — Manual Validation: 50-image spot check + failure analysis.

Samples 50 random images, prints their metadata for visual verification,
and runs failure analysis on the 5 evaluation queries.
"""

import os, sys, random, sqlite3
import pandas as pd
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_PATH = os.path.join(BASE, "data", "metadata.csv")
DB_PATH = os.path.join(BASE, "data", "glance.db")
IMAGE_DIR = os.path.join(BASE, "DATASET")


def spot_check(n=50):
    """Sample n random images and print metadata for visual verification."""
    df = pd.read_csv(META_PATH)
    sample = df.sample(n=n, random_state=42).reset_index(drop=True)

    print("=" * 110)
    print(f"SPOT CHECK — {n} random images")
    print("=" * 110)
    print(f"{'Idx':>3} {'File':<36} {'Category':<12} {'Color1':<10} {'Color2':<10} {'Color3':<10} "
          f"{'Location':<22} {'Style':<16} {'HasFile?':>8}")
    print("-" * 110)

    missing = 0
    for i, row in sample.iterrows():
        img_path = os.path.join(IMAGE_DIR, row["filename"])
        exists = os.path.isfile(img_path)
        if not exists:
            missing += 1
        print(f"{i:>3} {row['filename']:<36} {row['category']:<12} {row['color']:<10} "
              f"{str(row.get('color2','')):<10} {str(row.get('color3','')):<10} "
              f"{str(row.get('location','')):<22} {str(row.get('style','')):<16} "
              f"{'YES' if exists else 'MISSING':>8}")

    print("-" * 110)
    print(f"Images found: {n - missing}/{n}  |  Missing files: {missing}")


def failure_analysis():
    """Identify common failure patterns across all 5 queries."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM images", conn)
    conn.close()

    print("\n" + "=" * 110)
    print("FAILURE ANALYSIS")
    print("=" * 110)

    # 1. Color label sanity: check distribution
    print("\n1. Color distribution (top-10):")
    print(df["color"].value_counts().head(10).to_string())

    # 2. Color coherence: how often is the dominant color very low confidence?
    low_conf_color = (df["color_conf"] < 0.33).sum()
    print(f"\n2. Low-confidence color labels (conf < 0.33): {low_conf_color}/{len(df)} "
          f"({100 * low_conf_color / len(df):.1f}%)")

    # 3. Location distribution
    print("\n3. Location distribution:")
    print(df["location"].value_counts().to_string())

    # 4. Style distribution
    print("\n4. Style distribution:")
    print(df["style"].value_counts().to_string())

    # 5. Category distribution
    print("\n5. Category distribution:")
    print(df["category"].value_counts().to_string())

    # 6. Cross-tab: how many "shirt" items are labeled "park or garden"?
    shirt_park = len(df[(df["category"].str.contains("shirt", case=False)) &
                         (df["location"] == "park or garden")])
    print(f"\n6. Shirt + park-or-garden combos: {shirt_park}")

    # 7. Yellow coats
    yellow_coats = len(df[(df["category"] == "coat") & (df["color"] == "yellow")])
    yellow_coats_any = len(df[(df["category"] == "coat") &
                               ((df["color"] == "yellow") | (df["color2"] == "yellow") | (df["color3"] == "yellow"))])
    print(f"7. Yellow coats: {yellow_coats} (primary), {yellow_coats_any} (any color slot)")

    # 8. Formal + red/white shirts
    formal_red = len(df[(df["style"] == "formal style") &
                         (df["color"].isin(["red", "white"])) &
                         (df["category"].isin(["shirt", "suit", "blazer"]))])
    print(f"8. Formal-style + red/white shirt/suit/blazer: {formal_red}")

    # 9. Blue shirts
    blue_shirts = len(df[(df["category"].str.contains("shirt", case=False)) &
                          (df["color"] == "blue")])
    print(f"9. Blue shirts (any type): {blue_shirts}")

    print("\n" + "=" * 110)


if __name__ == "__main__":
    spot_check(50)
    failure_analysis()
