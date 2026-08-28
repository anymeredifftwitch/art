# scripts/test_clips_presets.py
"""
Presets de clips Twitch représentatifs pour tester le montage en conditions réelles :
- Preset 1 : Gaming avec Webcam (ex: GTA RP, Mario Kart, Only Up) -> Teste le SPLIT-SCREEN + Karaoké
- Preset 2 : Just Chatting avec Webcam (ex: Anecdote / Débat) -> Teste le FULLSCREEN ZOOM + Karaoké
- Preset 3 : Action / Clutch FPS -> Teste les FLASHS audio + SFX Whoosh
- Preset 4 : Moment Drôle / Rage -> Teste la détection du hook et des pics
"""

import sys
import os
import argparse

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Liste des clips types avec leurs caractéristiques
TEST_CLIPS = {
    "gaming_webcam_1": {
        "id": "CautiousBrightPuffinCoolCat-w4K_BwQh4c9yE-y3",
        "url": "https://clips.twitch.tv/CautiousBrightPuffinCoolCat-w4K_BwQh4c9yE-y3",
        "category": "Gaming + Webcam",
        "game": "Only Up!",
        "desc": "Moment de chute et rage en direct avec caméra dans le coin. Teste le Split-Screen et le hook.",
    },
    "gaming_webcam_2": {
        "id": "SmoothAgileDaikonKappaPride-1g8H-0s8p6K9",
        "url": "https://clips.twitch.tv/SmoothAgileDaikonKappaPride-1g8H-0s8p6K9",
        "category": "Gaming + Webcam (Action)",
        "game": "GTA V / RP",
        "desc": "Course poursuite avec la police. Teste le découpage Split-Screen et les sous-titres karaoké rapides.",
    },
    "chatting_webcam": {
        "id": "GentleBoldCaterpillarBrainSlug-1m9B-4k7_x9p",
        "url": "https://clips.twitch.tv/GentleBoldCaterpillarBrainSlug-1m9B-4k7_x9p",
        "category": "Just Chatting",
        "game": "Just Chatting",
        "desc": "Discussion face caméra avec le chat. Teste le Fullscreen Zoom centré et le badge titre fixe.",
    },
    "gaming_clutch": {
        "id": "BlushingTawnyDotterelFeelsBadMan-4n3k-9q1l",
        "url": "https://clips.twitch.tv/BlushingTawnyDotterelFeelsBadMan-4n3k-9q1l",
        "category": "FPS Clutch + Audio Peak",
        "game": "Counter-Strike 2",
        "desc": "Clutch sous tension avec pic de volume à la fin. Teste les flashs blancs et le Whoosh SFX.",
    },
}


def print_available_presets():
    print("\n📋 CLIPS DE TEST DISPONIBLES :")
    print("-" * 75)
    for key, data in TEST_CLIPS.items():
        print(f"🔹 [{key}] - {data['category']} ({data['game']})")
        print(f"   URL : {data['url']}")
        print(f"   Note: {data['desc']}")
        print("-" * 75)


def run_preset(preset_key=None, clip_url=None, no_upload=True, privacy="private"):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)
    import main

    if clip_url:
        target_ids = clip_url
    elif preset_key and preset_key in TEST_CLIPS:
        target_ids = TEST_CLIPS[preset_key]["url"]
    elif preset_key == "all":
        target_ids = ",".join([c["url"] for c in TEST_CLIPS.values()])
    else:
        print_available_presets()
        return

    print(f"\n🚀 Lancement du rendu pour : {target_ids}")
    print(f"Mode upload : {'DÉSACTIVÉ (Sauvegarde artefact locale/CI)' if no_upload else f'ACTIVÉ ({privacy})'}")

    main.PRIVACY_MODE = privacy
    main.NUMBER_OF_CLIPS_TO_ATTEMPT = 10
    main.main(
        clip_ids_input=target_ids,
        no_upload=no_upload,
        keep_files=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testeur de clips Twitch avec montage complet")
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(TEST_CLIPS.keys()) + ["all"],
        default=None,
        help="Nom du preset à exécuter (gaming_webcam_1, chatting_webcam, gaming_clutch, all)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL d'un clip Twitch spécifique",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Active l'upload sur YouTube (par défaut désactivé pour mode test)",
    )
    parser.add_argument(
        "--privacy",
        choices=["private", "unlisted", "public"],
        default="private",
        help="Visibilité YouTube si --upload est actif (défaut: private)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Affiche la liste des presets",
    )

    args = parser.parse_args()

    if args.list or (not args.preset and not args.url):
        print_available_presets()
    else:
        run_preset(
            preset_key=args.preset,
            clip_url=args.url,
            no_upload=not args.upload,
            privacy=args.privacy,
        )
