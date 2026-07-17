"""
Query Parser — extract structured attributes from natural language queries.

Uses keyword matching over known labels (colors, garments, locations, styles).
Returns a dict: {colors: [], garments: [], location: str | None, style: str | None}
"""

import re

# ---------------------------------------------------------------------------
# Known labels (must match metadata values)
# ---------------------------------------------------------------------------
KNOWN_COLORS = [
    "red", "green", "blue", "yellow", "white", "black", "pink", "purple",
    "orange", "brown", "gray", "navy", "beige", "cream", "maroon", "teal",
    "coral", "lavender", "olive", "mint", "gold", "silver", "taupe",
    "burgundy", "khaki", "indigo", "turquoise", "magenta", "charcoal", "ivory",
]

KNOWN_GARMENTS = [
    # dresses & one-pieces
    "dress", "jumpsuit", "romper", "bodysuit", "suit",
    # tops
    "shirt", "t-shirt", "blouse", "tank top",
    "sweater", "cardigan", "hoodie", "vest",
    # outerwear
    "jacket", "coat", "blazer", "raincoat",
    # bottoms
    "pants", "jeans", "shorts", "skirt",
    # accessories
    "scarf", "bag", "hat", "sunglasses", "belt", "tie",
    # footwear
    "sneakers", "shoes", "boots",
]

KNOWN_LOCATIONS = [
    "indoor setting", "office environment",
    "outdoor urban setting", "outdoor nature setting",
    "park or garden", "beach or coastal", "city street",
]

KNOWN_STYLES = [
    "casual style", "formal style", "sporty style",
    "bohemian style", "vintage style", "minimalist style",
]

# Short aliases for locations (query-friendly → stored label)
LOCATION_ALIASES = {
    "indoor": "indoor setting",
    "indoors": "indoor setting",
    "inside": "indoor setting",
    "office": "office environment",
    "urban": "outdoor urban setting",
    "city": "city street",
    "street": "city street",
    "nature": "outdoor nature setting",
    "outdoor": "outdoor nature setting",
    "outside": "outdoor nature setting",
    "park": "park or garden",
    "garden": "park or garden",
    "beach": "beach or coastal",
    "coastal": "beach or coastal",
    "sea": "beach or coastal",
    "ocean": "beach or coastal",
}

# Short aliases for styles
STYLE_ALIASES = {
    "casual": "casual style",
    "formal": "formal style",
    "sporty": "sporty style",
    "athletic": "sporty style",
    "bohemian": "bohemian style",
    "boho": "bohemian style",
    "vintage": "vintage style",
    "retro": "vintage style",
    "minimalist": "minimalist style",
    "minimal": "minimalist style",
    "business": "formal style",
    "professional": "formal style",
}


def parse_query(query: str) -> dict:
    """Parse a natural-language query into structured attributes.

    Returns dict with keys: colors, garments, location, style
    """
    q_lower = query.lower()
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]+", q_lower)

    # Colors — match in order, avoid substrings (e.g. "tan" in "tank")
    colors = []
    for c in KNOWN_COLORS:
        pattern = r"\b" + re.escape(c) + r"\b"
        if re.search(pattern, q_lower):
            if c not in colors:
                colors.append(c)

    # Garments — match in order
    garments = []
    for g in sorted(KNOWN_GARMENTS, key=len, reverse=True):  # longer first
        pattern = r"\b" + re.escape(g) + r"\b"
        if re.search(pattern, q_lower):
            if g not in garments:
                garments.append(g)

    # Location — check aliases first, then literal known locations
    location = None
    for alias, mapped in LOCATION_ALIASES.items():
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, q_lower):
            location = mapped
            break
    if location is None:
        for loc in KNOWN_LOCATIONS:
            pattern = r"\b" + re.escape(loc) + r"\b"
            if re.search(pattern, q_lower):
                location = loc
                break

    # Style — check aliases first
    style = None
    for alias, mapped in STYLE_ALIASES.items():
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, q_lower):
            style = mapped
            break
    if style is None:
        for sty in KNOWN_STYLES:
            pattern = r"\b" + re.escape(sty) + r"\b"
            if re.search(pattern, q_lower):
                style = sty
                break

    return {
        "colors": colors,
        "garments": garments,
        "location": location,
        "style": style,
    }


def format_parse(parsed: dict) -> str:
    parts = []
    if parsed["colors"]:
        parts.append(f"colors={parsed['colors']}")
    if parsed["garments"]:
        parts.append(f"garments={parsed['garments']}")
    if parsed["location"]:
        parts.append(f"location={parsed['location']}")
    if parsed["style"]:
        parts.append(f"style={parsed['style']}")
    return " | ".join(parts) if parts else "(none extracted)"


if __name__ == "__main__":
    queries = [
        "A person in a bright yellow raincoat.",
        "Professional business attire inside a modern office.",
        "Someone wearing a blue shirt sitting on a park bench.",
        "Casual weekend outfit for a city walk.",
        "A red tie and a white shirt in a formal setting.",
    ]
    for q in queries:
        parsed = parse_query(q)
        print(f"Q: {q}")
        print(f"  → {format_parse(parsed)}\n")
