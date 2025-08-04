import sys
import os
import json
from datetime import datetime, date

# Ajoute le dossier scripts au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

import get_top_clips
import download_clip
import generate_metadata
import upload_youtube
from classify_clip_type import classify_clip_type
from process_video_gameplay import process_gameplay_clip
from process_video_chatting import process_chatting_clip

# Dossiers et fichiers
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

PUBLISHED_HISTORY_FILE = os.path.join(DATA_DIR, 'published_shorts_history.json')
RAW_CLIP_PATH         = os.path.join(DATA_DIR, 'temp_raw_clip.mp4')
PROCESSED_CLIP_PATH   = os.path.join(DATA_DIR, 'temp_processed_short.mp4')

NUMBER_OF_CLIPS_TO_ATTEMPT_TO_PUBLISH = 3

def load_published_history():
    if not os.path.exists(PUBLISHED_HISTORY_FILE):
        return {}
    try:
        with open(PUBLISHED_HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

def save_published_history(history_data):
    with open(PUBLISHED_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, indent=2, ensure_ascii=False)

def get_today_published_ids(history_data):
    today_str = date.today().isoformat()
    return [item["twitch_clip_id"] for item in history_data.get(today_str, [])]

def add_to_history(history_data, clip_id, youtube_id):
    today_str = date.today().isoformat()
    if today_str not in history_data:
        history_data[today_str] = []
    history_data[today_str].append({
        "twitch_clip_id": clip_id,
        "youtube_short_id": youtube_id,
        "timestamp": datetime.now().isoformat()
    })

def main():
    history = load_published_history()
    today_published_ids = get_today_published_ids(history)

    clips_attempted = []
    twitch_token = get_top_clips.get_twitch_access_token()
    if not twitch_token:
        return

    eligible_clips = get_top_clips.get_eligible_short_clips(
        access_token=twitch_token,
        num_clips_per_source=50,
        days_ago=1,
        already_published_clip_ids=today_published_ids
    )
    if not eligible_clips:
        return

    published_count = 0
    for clip in eligible_clips:
        if published_count >= NUMBER_OF_CLIPS_TO_ATTEMPT_TO_PUBLISH:
            break

        if clip['id'] in clips_attempted or clip['id'] in today_published_ids:
            continue
        clips_attempted.append(clip['id'])

        downloaded_file = download_clip.download_twitch_clip(clip['url'], RAW_CLIP_PATH)
        if not downloaded_file:
            continue

        # Debug : quel game_name on analyse ?
        raw_game = clip.get('game_name')
        print(f"ℹ️  game_name brut du clip : {raw_game!r}")

        # Classification
        clip_type = classify_clip_type(clip)
        print(f"📂 Type de clip détecté : {clip_type}")

        # Choix du traitement
        if clip_type == "chatting":
            print("🛠️  Application du traitement JUST CHATTING")
            processed = process_chatting_clip(
                input_path=downloaded_file,
                output_path=PROCESSED_CLIP_PATH,
                max_duration_seconds=get_top_clips.MAX_VIDEO_DURATION_SECONDS,
                clip_data=clip
            )
        else:
            print("🛠️  Application du traitement GAMEPLAY")
            processed = process_gameplay_clip(
                input_path=downloaded_file,
                output_path=PROCESSED_CLIP_PATH,
                max_duration_seconds=get_top_clips.MAX_VIDEO_DURATION_SECONDS,
                clip_data=clip
            )

        if not processed:
            continue

        # Génération des métadonnées
        metadata = generate_metadata.generate_youtube_metadata(clip)

        # Upload YouTube activé
        try:
            youtube_service = upload_youtube.get_authenticated_service()
            video_id = upload_youtube.upload_youtube_short(
                youtube_service,
                processed,
                metadata
            )
            print(f"🎉 Short YouTube publié ! ID: {video_id}")
        except Exception as e:
            print(f"❌ Erreur lors de l'upload YouTube : {e}")
            video_id = None

        if video_id:
            add_to_history(history, clip['id'], video_id)
            save_published_history(history)
            today_published_ids = get_today_published_ids(history)
            published_count += 1

    print(f"\n🎉 {published_count} Short(s) traité(s) avec succès.")

if __name__ == "__main__":
    main()
