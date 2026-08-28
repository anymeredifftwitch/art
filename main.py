# main.py
# ============================================================
# Pipeline automatise de publication de Shorts YouTube
# depuis les meilleurs clips Twitch d'Anyme023.
#
# Flux par clip :
#   download -> classify -> analyze_audio + transcribe + detect_webcam
#              -> edit_short -> generate_metadata -> upload_youtube
# ============================================================

import sys
import os
import argparse
from datetime import datetime, date

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ajoute le dossier scripts au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

import get_top_clips
import download_clip
import generate_metadata
import upload_youtube
from classify_clip_type import classify_clip_type
from analyze_audio import analyze_audio
from generate_subtitles import transcribe
from detect_webcam import detect_webcam
from edit_short import edit_short

# Répertoires
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

PUBLISHED_HISTORY_FILE = os.path.join(DATA_DIR, 'published_shorts_history.json')

# Nombre de clips à tenter de publier par exécution
NUMBER_OF_CLIPS_TO_ATTEMPT = 2  # Ajustable

# Visibilite par defaut (override par --privacy)
PRIVACY_MODE = "public"


# ---------------------------------------------------------------------------
# Gestion de l'historique (format JSON)
# ---------------------------------------------------------------------------

def _load_published_history():
    import json

    if not os.path.exists(PUBLISHED_HISTORY_FILE):
        return {}
    try:
        with open(PUBLISHED_HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_published_history(history_data):
    import json

    with open(PUBLISHED_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, indent=2, ensure_ascii=False)


def _today_published_ids(history_data):
    today_str = date.today().isoformat()
    return [item["twitch_clip_id"] for item in history_data.get(today_str, [])]


def _add_to_history(history_data, clip_id, youtube_id):
    today_str = date.today().isoformat()
    if today_str not in history_data:
        history_data[today_str] = []
    history_data[today_str].append({
        "twitch_clip_id": clip_id,
        "youtube_short_id": youtube_id,
        "timestamp": datetime.now().isoformat(),
    })


# Mode d'upload / artifact
NO_UPLOAD_MODE = False
KEEP_FILES = False


def main(clip_ids_input=None, no_upload=False, keep_files=False):
    history = _load_published_history()
    already_published = _today_published_ids(history)

    # 1. Récupération des clips éligibles
    twitch_token = get_top_clips.get_twitch_access_token()
    if not twitch_token:
        print("❌ Impossible d'obtenir le token Twitch.")
        return

    if clip_ids_input:
        raw_ids = [c.strip() for c in clip_ids_input.split(",") if c.strip()]
        eligible_clips = get_top_clips.get_clips_by_ids(twitch_token, raw_ids)
    else:
        eligible_clips = get_top_clips.get_eligible_short_clips(
            access_token=twitch_token,
            num_clips_per_source=50,
            days_ago=30,
            already_published_clip_ids=already_published,
        )

    if not eligible_clips:
        print("Aucun clip éligible trouvé.")
        return

    published_count = 0
    attempted_ids = set(already_published) if not clip_ids_input else set()

    for clip in eligible_clips:
        if published_count >= NUMBER_OF_CLIPS_TO_ATTEMPT:
            break

        clip_id = clip.get('id')
        if clip_id in attempted_ids:
            continue
        attempted_ids.add(clip_id)

        print(f"\n--- Traitement du clip {clip_id} ---")
        print(f"Titre original : {clip.get('title')!r}")

        # Chemins temporaires
        raw_path = os.path.join(DATA_DIR, f"{clip_id}_raw.mp4")
        processed_path = os.path.join(DATA_DIR, f"{clip_id}_processed.mp4")

        # 2. Téléchargement
        if not download_clip.download_twitch_clip(clip['url'], raw_path):
            _cleanup(raw_path, processed_path)
            continue

        # 3. Classification (gameplay vs chatting)
        print(f"game_name brut du clip : {clip.get('game_name')!r}")
        clip_type = classify_clip_type(clip)
        clip["clip_type"] = clip_type
        print(f"Type de clip détecté : {clip_type}")

        # 4. Analyse audio (hook + pics)
        print("Analyse audio en cours...")
        audio = analyze_audio(raw_path, hook_duration=3.0)

        # 5. Transcription (Whisper FR)
        print("Transcription en cours...")
        subs = transcribe(raw_path)

        # 6. Détection webcam
        print("Détection webcam en cours...")
        webcam = detect_webcam(raw_path, num_samples=10)

        # 6b. Titre overlay (Groq ou heuristique, basé sur la transcription)
        overlay_title = generate_metadata.generate_video_title(clip, subs)
        clip["overlay_title"] = overlay_title

        # 7. Montage unifié
        print("Montage vidéo en cours...")
        edit_short(
            input_path=raw_path,
            output_path=processed_path,
            clip_data=clip,
            webcam_info=webcam,
            subtitles=subs,
            audio_analysis=audio,
            max_duration=get_top_clips.MAX_VIDEO_DURATION_SECONDS,
        )

        if not os.path.exists(processed_path):
            print("Le montage n'a pas produit de fichier de sortie.")
            _cleanup(raw_path, processed_path)
            continue

        # 8. Métadonnées (réutilise le titre overlay généré avant)
        metadata = generate_metadata.generate_youtube_metadata(clip, overlay_title)
        if PRIVACY_MODE != "public":
            metadata["privacyStatus"] = PRIVACY_MODE
            print(f"Mode TEST : privacyStatus = {PRIVACY_MODE}")

        # 9. Upload YouTube (ou sauvegarde artifact)
        if no_upload:
            print(f"💾 Mode ARTEFACT : Vidéo enregistrée dans {processed_path} (Upload ignoré)")
            _cleanup(raw_path)
            published_count += 1
            continue

        try:
            yt_service = upload_youtube.get_authenticated_service()
            video_id = upload_youtube.upload_youtube_short(
                yt_service, processed_path, metadata
            )
            print(f"Short YouTube publié ! ID: {video_id}")
        except Exception as exc:
            print(f"Erreur lors de l'upload YouTube : {exc}")
            video_id = None

        if video_id:
            _add_to_history(history, clip_id, video_id)
            _save_published_history(history)
            published_count += 1

        # Nettoyage
        if keep_files:
            _cleanup(raw_path)
        else:
            _cleanup(raw_path, processed_path)

    print(f"\n{published_count} Short(s) traité(s) avec succès.")


def _cleanup(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception as exc:
            print(f"Impossible de supprimer {p} : {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de publication de Shorts YouTube"
    )
    parser.add_argument(
        "--privacy",
        choices=["public", "private", "unlisted"],
        default="public",
        help="Visibilité du Short (défaut: public). Utiliser 'private' ou 'unlisted' pour un test.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=NUMBER_OF_CLIPS_TO_ATTEMPT,
        help=f"Nombre max de clips à publier (défaut: {NUMBER_OF_CLIPS_TO_ATTEMPT})",
    )
    parser.add_argument(
        "--clip-ids",
        type=str,
        default=None,
        help="ID(s) ou URL(s) de clips Twitch spécifiques séparés par des virgules (ex: ClipID1,ClipID2).",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Génère uniquement les vidéos MP4 sans tenter d'upload YouTube (idéal pour tester en artefact).",
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Conserve les vidéos MP4 générées dans le dossier data/ au lieu de les supprimer après upload.",
    )
    args = parser.parse_args()

    PRIVACY_MODE = args.privacy
    NUMBER_OF_CLIPS_TO_ATTEMPT = args.max_clips
    NO_UPLOAD_MODE = args.no_upload
    KEEP_FILES = args.keep_files

    main(
        clip_ids_input=args.clip_ids,
        no_upload=args.no_upload,
        keep_files=args.keep_files,
    )