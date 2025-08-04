# scripts/classify_clip_type.py

import os
import requests
from get_top_clips import get_twitch_access_token

CLIENT_ID        = os.getenv("TWITCH_CLIENT_ID")
HELIX_GAMES_URL  = "https://api.twitch.tv/helix/games"
HELIX_CLIPS_URL  = "https://api.twitch.tv/helix/clips"

def fetch_game_name(game_id, token):
    headers = {
        "Client-ID": CLIENT_ID,
        "Authorization": f"Bearer {token}"
    }
    resp = requests.get(HELIX_GAMES_URL, headers=headers, params={"id": game_id})
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if data:
        return data[0].get("name")
    return None

def fetch_game_id(clip_id, token):
    headers = {
        "Client-ID": CLIENT_ID,
        "Authorization": f"Bearer {token}"
    }
    resp = requests.get(HELIX_CLIPS_URL, headers=headers, params={"id": clip_id})
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if data:
        return data[0].get("game_id")
    return None

def classify_clip_type(clip_data):
    """
    Renvoie 'chatting' si Just Chatting, 'gameplay' sinon.
    """
    token = get_twitch_access_token()

    # 1️⃣ Game ID (depuis clip_data ou en fetchant)
    game_id = clip_data.get("game_id")
    print(f"🔍 Debug classify: clip_id={clip_data['id']}, initial game_id={game_id!r}")

    if not game_id:
        game_id = fetch_game_id(clip_data["id"], token)
        print(f"🔄 Game ID récupéré via API clips: {game_id!r}")

    # 2️⃣ Game Name
    game_name = None
    if game_id:
        game_name = fetch_game_name(game_id, token)
    print(f"🔍 Debug classify: final game_name={game_name!r}")

    # 3️⃣ Classification
    if game_name and game_name.lower() == "just chatting":
        return "chatting"
    elif not game_name:
        # Si pas de jeu détecté on considère chatting  
        return "chatting"
    else:
        return "gameplay"
