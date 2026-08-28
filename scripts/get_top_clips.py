import requests
import os
import sys
import json
from datetime import datetime, timedelta, timezone

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CLIENT_ID     = os.getenv("TWITCH_CLIENT_ID")
CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    print("❌ ERREUR: TWITCH_CLIENT_ID ou TWITCH_CLIENT_SECRET non définis.")
    sys.exit(1)

TWITCH_AUTH_URL = "https://id.twitch.tv/oauth2/token"
TWITCH_API_URL  = "https://api.twitch.tv/helix/clips"

TARGET_BROADCASTER_ID      = "737048563"
CLIP_LANGUAGE              = "fr"
MIN_VIDEO_DURATION_SECONDS = 15
MAX_VIDEO_DURATION_SECONDS = 180

def get_twitch_access_token():
    print("🔑 Récupération du jeton d'accès Twitch...")
    resp = requests.post(TWITCH_AUTH_URL, data={
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials"
    })
    resp.raise_for_status()
    token = resp.json()["access_token"]
    print("✅ Jeton d'accès Twitch récupéré.")
    return token

def fetch_clips(access_token, params):
    headers = {
        "Client-ID": CLIENT_ID,
        "Authorization": f"Bearer {access_token}"
    }
    resp = requests.get(TWITCH_API_URL, headers=headers, params=params)
    resp.raise_for_status()
    return resp.json().get("data", [])

def get_eligible_short_clips(access_token, num_clips_per_source=50, days_ago=90, already_published_clip_ids=None):
    if already_published_clip_ids is None:
        already_published_clip_ids = []

    end_date   = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_ago)
    seen       = set(already_published_clip_ids)
    all_clips  = []

    print(f"📊 Récupération des meilleurs clips pour Anyme023 (derniers {days_ago} jours)...")
    params = {
        "first": num_clips_per_source,
        "started_at": start_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "ended_at": end_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "broadcaster_id": TARGET_BROADCASTER_ID,
    }
    clips = fetch_clips(access_token, params)

    # Si aucun clip trouvé sur la période, chercher les top clips de tous les temps
    if not clips:
        print("ℹ️  Aucun clip récent trouvé, recherche des top clips globaux...")
        params_all = {
            "first": num_clips_per_source,
            "broadcaster_id": TARGET_BROADCASTER_ID,
        }
        clips = fetch_clips(access_token, params_all)

    for clip in clips:
        if clip.get("id") in seen:
            continue
        duration = float(clip.get("duration", 0.0))
        if not (MIN_VIDEO_DURATION_SECONDS <= duration <= MAX_VIDEO_DURATION_SECONDS):
            continue

        all_clips.append({
            "id": clip.get("id"),
            "url": clip.get("url"),
            "title": clip.get("title"),
            "broadcaster_name": clip.get("broadcaster_name"),
            "duration": duration,
            "language": clip.get("language"),
            "game_id": clip.get("game_id"),
            "game_name": clip.get("game_name"),
            "view_count": clip.get("view_count", 0),
        })
        seen.add(clip["id"])

    all_clips.sort(key=lambda x: x.get("view_count", 0), reverse=True)
    print(f"✅ Collecté {len(all_clips)} clip(s) éligible(s).")
    return all_clips


def get_clips_by_ids(access_token, clip_ids):
    """
    Récupère les informations détaillées de clips Twitch spécifiques par leur ID ou URL.
    """
    cleaned_ids = []
    for cid in clip_ids:
        cid = str(cid).strip()
        if not cid:
            continue
        if "/" in cid:
            cid = cid.rstrip("/").split("/")[-1].split("?")[0]
        cleaned_ids.append(cid)

    if not cleaned_ids:
        return []

    headers = {
        "Client-ID": CLIENT_ID,
        "Authorization": f"Bearer {access_token}"
    }
    params = [("id", cid) for cid in cleaned_ids]
    try:
        resp = requests.get(TWITCH_API_URL, headers=headers, params=params)
        resp.raise_for_status()
        raw_data = resp.json().get("data", [])
        results = []
        for clip in raw_data:
            duration = float(clip.get("duration", 0.0))
            results.append({
                "id": clip.get("id"),
                "url": clip.get("url"),
                "title": clip.get("title"),
                "broadcaster_name": clip.get("broadcaster_name"),
                "duration": duration,
                "language": clip.get("language"),
                "game_id": clip.get("game_id"),
                "game_name": clip.get("game_name")
            })
        if not results and cleaned_ids:
            print(f"ℹ️  Twitch API : métadonnées non trouvées, fallback direct pour téléchargement yt-dlp.")
            for cid in cleaned_ids:
                url = cid if cid.startswith("http") else f"https://clips.twitch.tv/{cid}"
                clip_id = cid.rstrip("/").split("/")[-1].split("?")[0]
                results.append({
                    "id": clip_id,
                    "url": url,
                    "title": "Anyme en Live",
                    "broadcaster_name": "anyme023",
                    "duration": 30.0,
                    "language": "fr",
                    "game_id": "",
                    "game_name": "Gameplay",
                })

        print(f"✅ {len(results)} clip(s) prêt(s) pour traitement.")
        return results
    except Exception as e:
        print(f"⚠️  Erreur API clips ({e}), création directe des clips de secours...")
        results = []
        for cid in cleaned_ids:
            url = cid if cid.startswith("http") else f"https://clips.twitch.tv/{cid}"
            clip_id = cid.rstrip("/").split("/")[-1].split("?")[0]
            results.append({
                "id": clip_id,
                "url": url,
                "title": "Anyme en Live",
                "broadcaster_name": "anyme023",
                "duration": 30.0,
                "language": "fr",
                "game_id": "",
                "game_name": "Gameplay",
            })
        return results

if __name__ == "__main__":
    token = get_twitch_access_token()
    clips = get_eligible_short_clips(token)
    print(clips[:2])  # debug

