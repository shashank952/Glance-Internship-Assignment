"""
Phase 2 — Pseudo-Label Generation: Location (Places365) + Style (CLIP).

Places365: ResNet50 pre-trained on Places365 → 365 scene categories → 7 location types.
Style: FashionCLIP zero-shot over 6 style prompts.
"""

import os, sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from transformers import CLIPProcessor, CLIPModel

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE, "data", "sample_images")
META_PATH = os.path.join(BASE, "data", "metadata.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Places365 → 7 location types mapping
# ---------------------------------------------------------------------------
# Maps Places365 scene category names to our condensed location labels
LOCATION_MAP = {
    # indoor setting
    "alcove": "indoor setting",
    "archive": "indoor setting",
    "art_gallery": "indoor setting",
    "art_studio": "indoor setting",
    "artists_loft": "indoor setting",
    "assembly_line": "indoor setting",
    "atrium/public": "indoor setting",
    "attic": "indoor setting",
    "auditorium": "indoor setting",
    "auto_factory": "indoor setting",
    "auto_showroom": "indoor setting",
    "bakery/shop": "indoor setting",
    "balcony/interior": "indoor setting",
    "ballroom": "indoor setting",
    "bank_vault": "indoor setting",
    "banquet_hall": "indoor setting",
    "bar": "indoor setting",
    "basement": "indoor setting",
    "bathroom": "indoor setting",
    "bazaar/indoor": "indoor setting",
    "beauty_salon": "indoor setting",
    "bedchamber": "indoor setting",
    "bedroom": "indoor setting",
    "beer_hall": "indoor setting",
    "bookstore": "indoor setting",
    "booth/indoor": "indoor setting",
    "bowling_alley": "indoor setting",
    "bus_interior": "indoor setting",
    "bus_station/indoor": "indoor setting",
    "butchers_shop": "indoor setting",
    "cafeteria": "indoor setting",
    "candy_store": "indoor setting",
    "car_interior": "indoor setting",
    "catacomb": "indoor setting",
    "chemistry_lab": "indoor setting",
    "childs_room": "indoor setting",
    "church/indoor": "indoor setting",
    "classroom": "indoor setting",
    "clean_room": "indoor setting",
    "closet": "indoor setting",
    "clothing_store": "indoor setting",
    "cockpit": "indoor setting",
    "coffee_shop": "indoor setting",
    "computer_room": "indoor setting",
    "conference_center": "indoor setting",
    "conference_room": "indoor setting",
    "corridor": "indoor setting",
    "delicatessen": "indoor setting",
    "department_store": "indoor setting",
    "dining_hall": "indoor setting",
    "dining_room": "indoor setting",
    "discotheque": "indoor setting",
    "dorm_room": "indoor setting",
    "dressing_room": "indoor setting",
    "drugstore": "indoor setting",
    "elevator/door": "indoor setting",
    "elevator_lobby": "indoor setting",
    "elevator_shaft": "indoor setting",
    "engine_room": "indoor setting",
    "entrance_hall": "indoor setting",
    "escalator/indoor": "indoor setting",
    "fabric_store": "indoor setting",
    "fastfood_restaurant": "indoor setting",
    "flea_market/indoor": "indoor setting",
    "florist_shop/indoor": "indoor setting",
    "food_court": "indoor setting",
    "galley": "indoor setting",
    "garage/indoor": "indoor setting",
    "general_store/indoor": "indoor setting",
    "gift_shop": "indoor setting",
    "greenhouse/indoor": "indoor setting",
    "gymnasium/indoor": "indoor setting",
    "hangar/indoor": "indoor setting",
    "hardware_store": "indoor setting",
    "home_office": "indoor setting",
    "home_theater": "indoor setting",
    "hospital": "indoor setting",
    "hospital_room": "indoor setting",
    "hotel_room": "indoor setting",
    "ice_cream_parlor": "indoor setting",
    "jail_cell": "indoor setting",
    "jewelry_shop": "indoor setting",
    "kindergarden_classroom": "indoor setting",
    "kitchen": "indoor setting",
    "laundromat": "indoor setting",
    "lecture_room": "indoor setting",
    "legislative_chamber": "indoor setting",
    "library/indoor": "indoor setting",
    "living_room": "indoor setting",
    "lobby": "indoor setting",
    "locker_room": "indoor setting",
    "market/indoor": "indoor setting",
    "mezzanine": "indoor setting",
    "movie_theater/indoor": "indoor setting",
    "museum/indoor": "indoor setting",
    "music_studio": "indoor setting",
    "natural_history_museum": "indoor setting",
    "nursery": "indoor setting",
    "nursing_home": "indoor setting",
    "operating_room": "indoor setting",
    "pantry": "indoor setting",
    "pet_shop": "indoor setting",
    "pharmacy": "indoor setting",
    "phone_booth": "indoor setting",
    "physics_laboratory": "indoor setting",
    "pizzeria": "indoor setting",
    "playroom": "indoor setting",
    "pub/indoor": "indoor setting",
    "reception": "indoor setting",
    "recreation_room": "indoor setting",
    "repair_shop": "indoor setting",
    "restaurant": "indoor setting",
    "restaurant_kitchen": "indoor setting",
    "sauna": "indoor setting",
    "server_room": "indoor setting",
    "shoe_shop": "indoor setting",
    "shopping_mall/indoor": "indoor setting",
    "shower": "indoor setting",
    "stage/indoor": "indoor setting",
    "staircase": "indoor setting",
    "storage_room": "indoor setting",
    "subway_station/platform": "indoor setting",
    "supermarket": "indoor setting",
    "sushi_bar": "indoor setting",
    "television_room": "indoor setting",
    "television_studio": "indoor setting",
    "throne_room": "indoor setting",
    "toyshop": "indoor setting",
    "train_interior": "indoor setting",
    "utility_room": "indoor setting",
    "veterinarians_office": "indoor setting",
    "waiting_room": "indoor setting",
    "wet_bar": "indoor setting",
    "youth_hostel": "indoor setting",
    # office environment
    "office": "office environment",
    "office_building": "office environment",
    "office_cubicles": "office environment",
    # outdoor urban setting
    "airfield": "outdoor urban setting",
    "airport_terminal": "outdoor urban setting",
    "alley": "outdoor urban setting",
    "amphitheater": "outdoor urban setting",
    "apartment_building/outdoor": "outdoor urban setting",
    "aqueduct": "outdoor urban setting",
    "archaelogical_excavation": "outdoor urban setting",
    "arena/hockey": "outdoor urban setting",
    "arena/performance": "outdoor urban setting",
    "arena/rodeo": "outdoor urban setting",
    "army_base": "outdoor urban setting",
    "athletic_field/outdoor": "outdoor urban setting",
    "auto_showroom": "outdoor urban setting",
    "balcony/exterior": "outdoor urban setting",
    "barn": "outdoor urban setting",
    "barndoor": "outdoor urban setting",
    "baseball_field": "outdoor urban setting",
    "basketball_court/indoor": "outdoor urban setting",
    "bazaar/outdoor": "outdoor urban setting",
    "bridge": "outdoor urban setting",
    "building_facade": "outdoor urban setting",
    "bullring": "outdoor urban setting",
    "bus_station/indoor": "outdoor urban setting",
    "cabin/outdoor": "outdoor urban setting",
    "campus": "outdoor urban setting",
    "carrousel": "outdoor urban setting",
    "castle": "outdoor urban setting",
    "cemetery": "outdoor urban setting",
    "chalet": "outdoor urban setting",
    "church/outdoor": "outdoor urban setting",
    "construction_site": "outdoor urban setting",
    "courthouse": "outdoor urban setting",
    "courtyard": "outdoor urban setting",
    "dam": "outdoor urban setting",
    "doorway/outdoor": "outdoor urban setting",
    "downtown": "outdoor urban setting",
    "driveway": "outdoor urban setting",
    "embassy": "outdoor urban setting",
    "excavation": "outdoor urban setting",
    "fire_escape": "outdoor urban setting",
    "fire_station": "outdoor urban setting",
    "football_field": "outdoor urban setting",
    "garage/outdoor": "outdoor urban setting",
    "gas_station": "outdoor urban setting",
    "general_store/outdoor": "outdoor urban setting",
    "hangar/outdoor": "outdoor urban setting",
    "heliport": "outdoor urban setting",
    "highway": "outdoor urban setting",
    "hotel/outdoor": "outdoor urban setting",
    "house": "outdoor urban setting",
    "industrial_area": "outdoor urban setting",
    "inn/outdoor": "outdoor urban setting",
    "junkyard": "outdoor urban setting",
    "kasbah": "outdoor urban setting",
    "kennel/outdoor": "outdoor urban setting",
    "landing_deck": "outdoor urban setting",
    "loading_dock": "outdoor urban setting",
    "lock_chamber": "outdoor urban setting",
    "mansion": "outdoor urban setting",
    "manufactured_home": "outdoor urban setting",
    "market/outdoor": "outdoor urban setting",
    "mausoleum": "outdoor urban setting",
    "medina": "outdoor urban setting",
    "motel": "outdoor urban setting",
    "museum/outdoor": "outdoor urban setting",
    "oast_house": "outdoor urban setting",
    "oilrig": "outdoor urban setting",
    "pagoda": "outdoor urban setting",
    "palace": "outdoor urban setting",
    "parking_garage/indoor": "outdoor urban setting",
    "parking_garage/outdoor": "outdoor urban setting",
    "parking_lot": "outdoor urban setting",
    "patio": "outdoor urban setting",
    "pavilion": "outdoor urban setting",
    "pier": "outdoor urban setting",
    "playground": "outdoor urban setting",
    "plaza": "outdoor urban setting",
    "porch": "outdoor urban setting",
    "promenade": "outdoor urban setting",
    "racecourse": "outdoor urban setting",
    "raceway": "outdoor urban setting",
    "railroad_track": "outdoor urban setting",
    "residential_neighborhood": "outdoor urban setting",
    "restaurant_patio": "outdoor urban setting",
    "roof_garden": "outdoor urban setting",
    "rope_bridge": "outdoor urban setting",
    "ruin": "outdoor urban setting",
    "runway": "outdoor urban setting",
    "schoolhouse": "outdoor urban setting",
    "science_museum": "outdoor urban setting",
    "shed": "outdoor urban setting",
    "shopfront": "outdoor urban setting",
    "ski_resort": "outdoor urban setting",
    "skyscraper": "outdoor urban setting",
    "slum": "outdoor urban setting",
    "soccer_field": "outdoor urban setting",
    "stable": "outdoor urban setting",
    "stadium/baseball": "outdoor urban setting",
    "stadium/football": "outdoor urban setting",
    "stadium/soccer": "outdoor urban setting",
    "stage/outdoor": "outdoor urban setting",
    "ticket_booth": "outdoor urban setting",
    "tower": "outdoor urban setting",
    "train_station/platform": "outdoor urban setting",
    "tree_house": "outdoor urban setting",
    "viaduct": "outdoor urban setting",
    "village": "outdoor urban setting",
    "volcano": "outdoor urban setting",
    "volleyball_court/outdoor": "outdoor urban setting",
    "water_tower": "outdoor urban setting",
    "wind_farm": "outdoor urban setting",
    "windmill": "outdoor urban setting",
    # outdoor nature setting
    "badlands": "outdoor nature setting",
    "bamboo_forest": "outdoor nature setting",
    "butte": "outdoor nature setting",
    "campsite": "outdoor nature setting",
    "canyon": "outdoor nature setting",
    "cliff": "outdoor nature setting",
    "corn_field": "outdoor nature setting",
    "corral": "outdoor nature setting",
    "creek": "outdoor nature setting",
    "crevasse": "outdoor nature setting",
    "desert/sand": "outdoor nature setting",
    "desert/vegetation": "outdoor nature setting",
    "desert_road": "outdoor nature setting",
    "excavation": "outdoor nature setting",
    "farm": "outdoor nature setting",
    "field/cultivated": "outdoor nature setting",
    "field/wild": "outdoor nature setting",
    "field_road": "outdoor nature setting",
    "forest/broadleaf": "outdoor nature setting",
    "forest_path": "outdoor nature setting",
    "forest_road": "outdoor nature setting",
    "glacier": "outdoor nature setting",
    "grotto": "outdoor nature setting",
    "hayfield": "outdoor nature setting",
    "hunting_lodge/outdoor": "outdoor nature setting",
    "ice_floe": "outdoor nature setting",
    "ice_shelf": "outdoor nature setting",
    "iceberg": "outdoor nature setting",
    "islet": "outdoor nature setting",
    "lagoon": "outdoor nature setting",
    "lake/natural": "outdoor nature setting",
    "landfill": "outdoor nature setting",
    "marsh": "outdoor nature setting",
    "mountain": "outdoor nature setting",
    "mountain_path": "outdoor nature setting",
    "mountain_snowy": "outdoor nature setting",
    "orchard": "outdoor nature setting",
    "pasture": "outdoor nature setting",
    "pond": "outdoor nature setting",
    "raft": "outdoor nature setting",
    "rainforest": "outdoor nature setting",
    "rice_paddy": "outdoor nature setting",
    "river": "outdoor nature setting",
    "rock_arch": "outdoor nature setting",
    "snowfield": "outdoor nature setting",
    "swamp": "outdoor nature setting",
    "swimming_hole": "outdoor nature setting",
    "tree_farm": "outdoor nature setting",
    "trench": "outdoor nature setting",
    "tundra": "outdoor nature setting",
    "valley": "outdoor nature setting",
    "vegetable_garden": "outdoor nature setting",
    "vineyard": "outdoor nature setting",
    "waterfall": "outdoor nature setting",
    "watering_hole": "outdoor nature setting",
    "wheat_field": "outdoor nature setting",
    # park or garden
    "botanical_garden": "park or garden",
    "formal_garden": "park or garden",
    "fountain": "park or garden",
    "gazebo/exterior": "park or garden",
    "japanese_garden": "park or garden",
    "lawn": "park or garden",
    "park": "park or garden",
    "picnic_area": "park or garden",
    "sandbox": "park or garden",
    "topiary_garden": "park or garden",
    "water_park": "park or garden",
    "yard": "park or garden",
    "zen_garden": "park or garden",
    # beach or coastal
    "beach": "beach or coastal",
    "beach_house": "beach or coastal",
    "boardwalk": "beach or coastal",
    "boat_deck": "beach or coastal",
    "boathouse": "beach or coastal",
    "canal/natural": "beach or coastal",
    "canal/urban": "beach or coastal",
    "coast": "beach or coastal",
    "fishpond": "beach or coastal",
    "harbor": "beach or coastal",
    "hot_spring": "beach or coastal",
    "lighthouse": "beach or coastal",
    "moat/water": "beach or coastal",
    "ocean": "beach or coastal",
    "swimming_pool/indoor": "beach or coastal",
    "swimming_pool/outdoor": "beach or coastal",
    "underwater/ocean_deep": "beach or coastal",
    "wave": "beach or coastal",
    # city street
    "crosswalk": "city street",
    "street": "city street",
}

# ---------------------------------------------------------------------------
# Style prompts — unchanged (CLIP zero-shot)
# ---------------------------------------------------------------------------
STYLE_PROMPTS = [
    "casual style",
    "formal style",
    "sporty style",
    "bohemian style",
    "vintage style",
    "minimalist style",
]

# ---------------------------------------------------------------------------
# Places365 preprocessing
# ---------------------------------------------------------------------------
PLACES_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_places365():
    """Load ResNet50 with Places365 weights."""
    print(f"[Places365] Loading model…")
    model = models.resnet50(weights=None, num_classes=365)
    url = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth"
    state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")["state_dict"]
    # Remove 'module.' prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def classify_location(model, image_path):
    """Return (location_label, confidence) using Places365."""
    img = Image.open(image_path).convert("RGB")
    input_tensor = PLACES_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    best_idx = int(np.argmax(probs))
    best_prob = float(probs[best_idx])

    # Get Places365 category name from index
    # The category list (sorted by index)
    # We stored it in our LOCATION_MAP keys
    # Load names from url
    names_url = "https://raw.githubusercontent.com/zhoubolei/places_devkit/master/categories_places365.txt"
    import urllib.request
    try:
        resp = urllib.request.urlopen(names_url)
        lines = resp.read().decode("utf-8").strip().splitlines()
    except Exception:
        # Fallback: use short hardcoded list
        return "indoor setting", best_prob

    # Parse "name index" → name (strip /a/, /b/, etc. prefix)
    cat_name = None
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[-1] == str(best_idx):
            raw = " ".join(parts[:-1])  # e.g. "/a/airfield"
            cat_name = raw.split("/", 2)[-1]  # "airfield"
            break

    if cat_name and cat_name in LOCATION_MAP:
        return LOCATION_MAP[cat_name], best_prob
    # Fallback: check partial matches
    if cat_name:
        for key, val in LOCATION_MAP.items():
            if key in cat_name or cat_name in key:
                return val, best_prob
    return "indoor setting", best_prob


def main():
    df = pd.read_csv(META_PATH)
    print(f"[Phase 2] Loaded {len(df)} rows from {META_PATH}")

    # Load Places365 for location
    loc_model = load_places365()

    # Load CLIP for style
    print(f"[Style] Loading CLIP on {DEVICE}…")
    clip = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    style_inputs = processor(text=STYLE_PROMPTS, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        style_text = clip.text_model(**style_inputs)
        style_embeds = clip.text_projection(style_text.pooler_output)
        style_embeds = style_embeds / style_embeds.norm(dim=-1, keepdim=True)

    image_dir = IMAGE_DIR
    locations, loc_confs = [], []
    styles, style_confs = [], []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Location + Style"):
        img_path = os.path.join(image_dir, row["filename"])
        if not os.path.isfile(img_path):
            locations.append("unknown")
            loc_confs.append(0.0)
            styles.append("unknown")
            style_confs.append(0.0)
            continue

        # Location via Places365
        try:
            loc, loc_c = classify_location(loc_model, img_path)
        except Exception as e:
            loc, loc_c = "unknown", 0.0
            print(f"\n  [WARN] Location failed for {row['filename']}: {e}")

        # Style via CLIP zero-shot
        try:
            image = Image.open(img_path).convert("RGB")
            style_inputs_img = processor(images=image, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                vision_out = clip.vision_model(**style_inputs_img)
                img_emb = clip.visual_projection(vision_out.pooler_output)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

            sims = (img_emb @ style_embeds.T).squeeze(0).cpu().numpy()
            exp_s = np.exp(sims - sims.max())
            probs = exp_s / exp_s.sum()
            best_idx = int(np.argmax(sims))
            sty = STYLE_PROMPTS[best_idx]
            sty_c = float(probs[best_idx])
        except Exception as e:
            sty, sty_c = "unknown", 0.0
            print(f"\n  [WARN] Style failed for {row['filename']}: {e}")

        locations.append(loc)
        loc_confs.append(loc_c)
        styles.append(sty)
        style_confs.append(sty_c)

    df["location"] = locations
    df["loc_conf"] = loc_confs
    df["style"] = styles
    df["style_conf"] = style_confs

    df.to_csv(META_PATH, index=False)
    print(f"\n[Phase 2] Saved {len(df)} rows to {META_PATH}")
    print("\nLocation distribution:")
    print(df["location"].value_counts().to_string())
    print("\nStyle distribution:")
    print(df["style"].value_counts().to_string())
    print(f"\nColumns now: {list(df.columns)}")


if __name__ == "__main__":
    main()
