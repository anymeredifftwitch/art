# scripts/classify_clip_type.py

import os
import requests
from get_top_clips import get_twitch_access_token

HELIX_GAMES_URL  = "https://api.twitch.tv/helix/games"
HELIX_CLIPS_URL  = "https://api.twitch.tv/helix/clips"

def fetch_game_name(game_id, token):
    client_id = os.getenv("TWITCH_CLIENT_ID")
    if not token or not client_id or not game_id:
        return None
    headers = {
        "Client-ID": client_id,
        "Authorization": f"Bearer {token}"
    }
    try:
        resp = requests.get(HELIX_GAMES_URL, headers=headers, params={"id": game_id}, timeout=5)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data:
            return data[0].get("name")
    except Exception as e:
        print(f"⚠️  Erreur fetch_game_name ({e})")
    return None

def fetch_game_id(clip_id, token):
    client_id = os.getenv("TWITCH_CLIENT_ID")
    if not token or not client_id or not clip_id:
        return None
    headers = {
        "Client-ID": client_id,
        "Authorization": f"Bearer {token}"
    }
    try:
        resp = requests.get(HELIX_CLIPS_URL, headers=headers, params={"id": clip_id}, timeout=5)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data:
            return data[0].get("game_id")
    except Exception as e:
        print(f"⚠️  Erreur fetch_game_id ({e})")
    return None

def classify_clip_type(clip_data):
    """
    Renvoie 'chatting' si Just Chatting, 'gameplay' sinon.
    """
    # Si le nom du jeu est déjà fourni dans clip_data
    game_name = clip_data.get("game_name")
    if game_name:
        if game_name.strip().lower() in ["just chatting", "discussion", "talk shows & podcasts"]:
            return "chatting"
        return "gameplay"

    try:
        token = get_twitch_access_token()
    except Exception:
        token = None

    # 1️⃣ Game ID (depuis clip_data ou en fetchant)
    game_id = clip_data.get("game_id")
    if not game_id and token:
        game_id = fetch_game_id(clip_data.get("id"), token)

    # 2️⃣ Game Name
    if game_id and token:
        game_name = fetch_game_name(game_id, token)

    # 3️⃣ Classification
    if game_name and game_name.lower() in ["just chatting", "discussion", "talk shows & podcasts"]:
        return "chatting"
    elif not game_name:
        # Par défaut si aucun jeu identifié
        return "gameplay"
    else:
        return "gameplay"
