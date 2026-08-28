#!/usr/bin/env python3
# test_generate_titles.py
"""
Script de test et d'itération pour la génération de titres avec l'IA Groq.
Permet d'évaluer la qualité des titres générés sur 10 clips Twitch en conditions réelles
SANS télécharger les vidéos ni faire de montage.
"""

import os
import sys
import json
import argparse
import re
from datetime import datetime
from dotenv import load_dotenv

# Forcer UTF-8 sur Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Charger les variables d'environnement (.env)
load_dotenv()

# Ajouter le dossier scripts au PATH
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
import generate_metadata

# 10 NOUVEAUX clips réels d'Anyme023 pour tester la généralisation
DEFAULT_BENCHMARK_CLIPS = [
    {
        "id": "LaconicIronicAsparagusStinkyCheese-rBu1RqmHGKynrUyv",
        "url": "https://www.twitch.tv/anyme023/clip/LaconicIronicAsparagusStinkyCheese-rBu1RqmHGKynrUyv",
        "title": "Course poursuite de 45 minutes avec la police",
        "broadcaster_name": "Anyme023",
        "game_name": "GTA V",
        "duration": 42.0,
        "transcription": "Accélère, tourne à droite dans la ruelle ! Y'a 5 voitures de flics derrière nous, si on se fait choper c'est direct 10 ans de prison fédérale !"
    },
    {
        "id": "RespectfulDreamyGarageOSfrog-ykxWE3DUTmLShF6O",
        "url": "https://www.twitch.tv/anyme023/clip/RespectfulDreamyGarageOSfrog-ykxWE3DUTmLShF6O",
        "title": "Pack opening des 500k crédits, la déception totale",
        "broadcaster_name": "Anyme023",
        "game_name": "EA Sports FC 25",
        "duration": 55.0,
        "transcription": "Allez s'il te plaît une animation... France... BU... MAIS NON C'EST ENCORE UN 82 GÉNÉRAL ! J'ai jeté 100 balles à la poubelle !"
    },
    {
        "id": "RelentlessWildCockroachThisIsSparta-5jXDni3DNyIDy-Ll",
        "url": "https://www.twitch.tv/anyme023/clip/RelentlessWildCockroachThisIsSparta-5jXDni3DNyIDy-Ll",
        "title": "4 heures de montée pour ça... je désinstalle",
        "broadcaster_name": "Anyme023",
        "game_name": "Only Up!",
        "duration": 31.0,
        "transcription": "Je glisse sur le tuyau... non non non raccroche-toi ! TOUT EN BAS ! Je suis revenu au tout début ! J'éteins le stream, bonne soirée."
    },
    {
        "id": "FaithfulSuspiciousSalmonKAPOW-qyWvDyDaM7yFE00J",
        "url": "https://www.twitch.tv/anyme023/clip/FaithfulSuspiciousSalmonKAPOW-qyWvDyDaM7yFE00J",
        "title": "Il m'appelle sur Discord pour me faire une demande en mariage",
        "broadcaster_name": "Anyme023",
        "game_name": "Just Chatting",
        "duration": 48.0,
        "transcription": "Le mec monte sur Discord, commence à réciter un poème d'amour de 3 pages et sort une bague en plastique devant 4000 personnes..."
    },
    {
        "id": "HelplessBrightSandwichChocolateRain-mpCW1LjVccck_l7C",
        "url": "https://www.twitch.tv/anyme023/clip/HelplessBrightSandwichChocolateRain-mpCW1LjVccck_l7C",
        "title": "Mort au jour 98 en mode Hardcore",
        "broadcaster_name": "Anyme023",
        "game_name": "Minecraft",
        "duration": 26.0,
        "transcription": "Je mine tranquillement mon diamant, et là un creeper tombe du plafond sans faire un bruit... Jour 98, mon monde est supprimé."
    },
    {
        "id": "TenderTameSalamanderKAPOW-mX92kLqP410",
        "url": "https://www.twitch.tv/anyme023/clip/TenderTameSalamanderKAPOW-mX92kLqP410",
        "title": "Le vol du siècle à 1 mètre de la ligne d'arrivée",
        "broadcaster_name": "Anyme023",
        "game_name": "Mario Kart 8 Deluxe",
        "duration": 33.0,
        "transcription": "Dernier tour, je suis 1er avec 10 secondes d'avance... Carapace bleue, éclair, carapace rouge ! Je finis 8ème !"
    },
    {
        "id": "ProudGracefulPigeonKappaClaus-kLm890v",
        "url": "https://www.twitch.tv/anyme023/clip/ProudGracefulPigeonKappaClaus-kLm890v",
        "title": "Ce que le livreur m'a apporté à 3h du matin",
        "broadcaster_name": "Anyme023",
        "game_name": "Just Chatting",
        "duration": 50.0,
        "transcription": "J'ai commandé deux burgers et des frites. J'ouvre le sac, y'a juste deux sauces mayo et une boîte vide avec un mot écrit dessus !"
    },
    {
        "id": "HilariousGentleLlamaPogChamp-zZ49021",
        "url": "https://www.twitch.tv/anyme023/clip/HilariousGentleLlamaPogChamp-zZ49021",
        "title": "Ils ont décollé avec le vaisseau sans moi",
        "broadcaster_name": "Anyme023",
        "game_name": "Lethal Company",
        "duration": 29.0,
        "transcription": "Les gars ouvrez la porte ! Le monstre arrive ! Pourquoi le vaisseau décolle ?! Vous m'avez laissé crever comme une merde !"
    },
    {
        "id": "GloriousQuickCheetahTriHard-aBc1234",
        "url": "https://www.twitch.tv/anyme023/clip/GloriousQuickCheetahTriHard-aBc1234",
        "title": "En mode vitesse max les yeux fermés",
        "broadcaster_name": "Anyme023",
        "game_name": "Beat Saber",
        "duration": 35.0,
        "transcription": "Mes mains bougent toutes seules, combo x8, full combo en mode expert+ ! Même moi je sais pas comment j'ai fait ça !"
    },
    {
        "id": "SillyWildWolfBibleThump-qWe5678",
        "url": "https://www.twitch.tv/anyme023/clip/SillyWildWolfBibleThump-qWe5678",
        "title": "Ma réponse à la question de géographie va me hanter",
        "broadcaster_name": "Anyme023",
        "game_name": "Just Chatting",
        "duration": 41.0,
        "transcription": "Question : Quelle est la capitale de l'Australie ? Moi en toute confiance : 'Sydney évidemment !'... Tout le chat s'est foutu de ma gueule pendant 1 heure."
    }
]


def _get_groq_candidate_models(client=None, preferred_model=None):
    """
    Construit une liste ordonnée et exhaustive de modèles Groq à tester en cascade.
    Interroge également l'API Groq pour inclure dynamiquement tous les modèles de chat disponibles.
    """
    preferred = [
        preferred_model,
        # 1. Modèles Llama 3.3 / 3.1 haute performance
        "llama-3.3-70b-versatile",
        "llama-3.3-70b-specdec",
        "llama-3.1-8b-instant",
        # 2. Modèles DeepSeek
        "deepseek-r1-distill-llama-70b",
        "deepseek-r1-distill-qwen-32b",
        # 3. Modèles Qwen
        "qwen-2.5-32b",
        "qwen-2.5-coder-32b",
        # 4. Modèles Google Gemma
        "gemma2-9b-it",
        # 5. Modèles Llama 3.2
        "llama-3.2-11b-vision-preview",
        "llama-3.2-3b-preview",
        "llama-3.2-1b-preview",
        # 6. Modèles Llama 3 legacy
        "llama3-70b-8192",
        "llama3-8b-8192",
    ]
    models = [m for m in preferred if m]

    if client:
        try:
            available = client.models.list()
            # Filtrer les modèles spécialisés non-chat
            exclude_keywords = ["whisper", "embed", "tts", "guard", "audio"]
            for item in available.data:
                mid = getattr(item, "id", "")
                if mid and not any(k in mid.lower() for k in exclude_keywords):
                    if mid not in models:
                        models.append(mid)
        except Exception:
            pass

    return list(dict.fromkeys(models))


def call_groq_completion(prompt, api_key, model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=250):
    """Effectue un appel direct à Groq avec gestion des modèles de secours."""
    try:
        from groq import Groq
    except ImportError:
        print("❌ La bibliothèque 'groq' n'est pas installée. Exécutez : pip install groq")
        return None

    client = Groq(api_key=api_key)
    models_to_try = _get_groq_candidate_models(client=client, preferred_model=model)

    for m in models_to_try:
        try:
            resp = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": "Tu es un expert en copywriting viral YouTube Shorts et TikTok francophone."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ Modèle Groq '{m}' indisponible ({e}), essai du modèle suivant...")
            continue
    return None


def generate_titles_for_clip(clip, api_key, model="llama-3.3-70b-versatile"):
    """
    Génère l'ensemble des titres pour un clip donné :
    1. Titre YouTube Shorts principal
    2. Titre Overlay court (haut de vidéo)
    3. Variantes de styles (Storytelling, Drama, POV)
    4. Hashtags
    """
    title_raw = clip.get("title", "")
    streamer = clip.get("broadcaster_name", "Anyme023")
    game = clip.get("game_name") or "Gaming"
    transcription = clip.get("transcription", "")

    if not api_key:
        # Fallback heuristique si pas de clé Groq
        overlay = generate_metadata._heuristic_video_title(transcription, title_raw)
        yt_title = generate_metadata._heuristic_title(clip)
        hashtags = generate_metadata._heuristic_hashtags(clip)
        return {
            "overlay_title": overlay,
            "youtube_title": yt_title,
            "variantes": [yt_title],
            "hashtags": hashtags,
            "source": "heuristic_fallback"
        }

    # Prompt complet pour générer titres et variantes en une seule requête Groq
    prompt = f"""Tu es un expert d'élite en création de titres viraux pour YouTube Shorts et TikTok (audience Gaming/Twitch FR).

Données du clip :
- Streamer : {streamer}
- Jeu / Catégorie : {game}
- Titre original Twitch : << {title_raw} >>
- Extrait audio / Contexte : << {transcription} >>

STYLE & TON EXIGÉS :
- Langage direct, familier, drôle, sans filtre, percutant, style commu Twitch FR / TikTok.
- Exemples de structures à privilégier :
  * 'POV: Anyme [action choc] en LIVE' (ex: POV: Anyme pète son crâne en LIVE, POV: Anyme FOU LE FEU EN LIVE)
  * 'POV: IL [action choc]' (ex: POV: IL SE CHIE DESSUS DE PEUR)
  * Phrase choc en MAJUSCULES (ex: IL A DIT QUOI LA ???, LES VIEWERS SONT FOUS)
  * 'Anyme [action] et [punchline]' (ex: Anyme CLUTCH et se PISSE DESSUS, ANYME CLAQUE UN SMIC SUR CS2)
- Mots clés en MAJUSCULES pour l'emphase.
- AUCUN emoji, AUCUN hashtag dans les titres.

EXEMPLES DE CE QU'ON VEUT EXACTEMENT :
- POV: Anyme pète son crâne en LIVE
- POV: Anyme raconte sa PIRE ANECDOTE en LIVE
- IL A DIT QUOI LA ???
- POV: Anyme FOU LE FEU EN LIVE
- LES VIEWERS SONT FOUS
- Anyme CLUTCH et se PISSE DESSUS
- POV: IL SE CHIE DESSUS DE PEUR
- ANYME CLAQUE UN SMIC SUR CS2
- POV ANYME DRAGUE UN TRANS EN LIVE
- POV: ANYME SE FAIL et PLEURE

Consignes :
1. OVERLAY : Titre court (20 à 42 caractères max) pour le haut de l'écran. Ultra percutant.
2. TITRE_YOUTUBE : Titre principal YouTube Shorts (max 85 car), commence par "Anyme | ".
3. VARIANTE_DRAMA : Alternative plus dramatique/punchline (commence par "Anyme | ", sans emoji).
4. VARIANTE_POV : Alternative style POV ou situation (commence par "Anyme | ", sans emoji).
5. HASHTAGS : 5 hashtags pertinents (#Anyme #TwitchFR ...).

Réponds UNIQUEMENT dans ce format exact :
OVERLAY: <titre overlay>
TITRE_YOUTUBE: <titre youtube>
VARIANTE_DRAMA: <variante drama>
VARIANTE_POV: <variante pov>
HASHTAGS: #tag1 #tag2 #tag3 #tag4 #tag5
"""

    content = call_groq_completion(prompt, api_key, model=model, temperature=0.85)

    if not content:
        overlay = generate_metadata._heuristic_video_title(transcription, title_raw)
        yt_title = generate_metadata._heuristic_title(clip)
        hashtags = generate_metadata._heuristic_hashtags(clip)
        return {
            "overlay_title": overlay,
            "youtube_title": yt_title,
            "variantes": [yt_title],
            "hashtags": hashtags,
            "source": "heuristic_fallback_error"
        }

    # Parser les champs
    overlay_match = re.search(r"OVERLAY:\s*(.+?)(?:\n|$)", content)
    yt_match = re.search(r"TITRE_YOUTUBE:\s*(.+?)(?:\n|$)", content)
    drama_match = re.search(r"VARIANTE_DRAMA:\s*(.+?)(?:\n|$)", content)
    pov_match = re.search(r"VARIANTE_POV:\s*(.+?)(?:\n|$)", content)
    hashtags_match = re.search(r"HASHTAGS:\s*(.+?)(?:\n|$)", content)

    overlay = overlay_match.group(1).strip() if overlay_match else ""
    overlay = generate_metadata._clean_title(overlay)
    if len(overlay) > 45:
        overlay = overlay[:42].strip() + "..."

    yt_title = yt_match.group(1).strip() if yt_match else f"Anyme | {title_raw}"
    yt_title = generate_metadata._clean_title(yt_title)

    v_drama = drama_match.group(1).strip() if drama_match else ""
    v_pov = pov_match.group(1).strip() if pov_match else ""

    variantes = [yt_title]
    if v_drama and v_drama != yt_title:
        variantes.append(generate_metadata._clean_title(v_drama))
    if v_pov and v_pov != yt_title and v_pov != v_drama:
        variantes.append(generate_metadata._clean_title(v_pov))

    hashtags_raw = hashtags_match.group(1).strip() if hashtags_match else ""
    hashtags = [t.strip() for t in hashtags_raw.split() if t.startswith("#")]

    return {
        "overlay_title": overlay,
        "youtube_title": yt_title,
        "variantes": variantes,
        "hashtags": hashtags,
        "source": "groq"
    }


def fetch_live_twitch_clips(limit=10, days=30):
    """Tente de récupérer les clips récents directement via l'API Twitch."""
    try:
        import get_top_clips
        token = get_top_clips.get_twitch_access_token()
        if not token:
            return None
        clips = get_top_clips.get_eligible_short_clips(
            access_token=token,
            num_clips_per_source=limit * 2,
            days_ago=days
        )
        return clips[:limit]
    except Exception as e:
        print(f"⚠️ Impossible de récupérer les clips Twitch en direct : {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Test de génération de titres IA Groq")
    parser.add_argument("--groq-key", type=str, default=None, help="Clé API Groq (sinon utilise GROQ_API_KEY du .env)")
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile", help="Modèle Groq")
    parser.add_argument("--twitch", action="store_true", help="Récupérer 10 clips en direct depuis l'API Twitch")
    parser.add_argument("--days", type=int, default=30, help="Nombre de jours pour la recherche Twitch")
    parser.add_argument("--output-md", type=str, default="test_titles_results.md", help="Fichier de sortie Markdown")
    parser.add_argument("--output-json", type=str, default="test_titles_results.json", help="Fichier de sortie JSON")
    args = parser.parse_args()

    groq_api_key = args.groq_key or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("\n⚠️ AVERTISSEMENT : Aucune clé GROQ_API_KEY trouvée dans l'environnement ni dans .env.")
        print("Les titres seront générés via le système de fallback heuristique.")
        print("Pour activer Groq, ajoutez GROQ_API_KEY=gsk_... dans votre fichier .env ou passez --groq-key.\n")
    else:
        print(f"🔑 Clé Groq détectée. Modèle sélectionné : {args.model}")

    # Récupération des clips (Live Twitch ou Benchmark)
    clips = None
    if args.twitch:
        print("🌐 Récupération des clips en direct depuis Twitch...")
        clips = fetch_live_twitch_clips(limit=10, days=args.days)

    if not clips:
        print("📌 Utilisation du jeu de test de 10 clips Twitch d'Anyme023.")
        clips = DEFAULT_BENCHMARK_CLIPS

    print(f"\n🚀 Lancement du test de génération sur {len(clips)} clips...\n")

    results = []
    for idx, clip in enumerate(clips, 1):
        print(f"[{idx}/{len(clips)}] Traitement du clip : {clip.get('title')[:40]}...")
        gen_data = generate_titles_for_clip(clip, groq_api_key, model=args.model)
        item = {
            "index": idx,
            "clip_id": clip.get("id"),
            "url": clip.get("url"),
            "original_title": clip.get("title"),
            "game_name": clip.get("game_name"),
            "transcription": clip.get("transcription", ""),
            "generated_youtube_title": gen_data["youtube_title"],
            "generated_overlay_title": gen_data["overlay_title"],
            "variations": gen_data["variantes"],
            "hashtags": gen_data["hashtags"],
            "source": gen_data["source"]
        }
        results.append(item)

    # Sauvegarde JSON
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Création du rapport Markdown
    md_content = f"# 🎬 Rapport de Test de Génération de Titres (Groq IA)\n\n"
    md_content += f"- **Date** : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    md_content += f"- **Modèle** : `{args.model}`\n"
    md_content += f"- **Nombre de clips testés** : {len(results)}\n\n"
    md_content += "---\n\n"

    for item in results:
        md_content += f"### #{item['index']} — [{item['original_title']}]({item['url']})\n"
        md_content += f"- 🔗 **Lien Twitch** : {item['url']}\n"
        md_content += f"- 🎮 **Jeu / Catégorie** : {item['game_name'] or 'Non spécifié'}\n"
        if item['transcription']:
            md_content += f"- 🗣️ **Contexte audio** : *\"{item['transcription']}\"*\n"
        md_content += f"- 🔴 **Titre YouTube Shorts généré** : **`{item['generated_youtube_title']}`**\n"
        md_content += f"- 📺 **Titre Overlay Vidéo (Haut)** : `{item['generated_overlay_title']}`\n"
        if len(item['variations']) > 1:
            md_content += f"- 💡 **Variantes de titres proposées** :\n"
            for v in item['variations']:
                md_content += f"  - `{v}`\n"
        md_content += f"- 🏷️ **Hashtags** : `{' '.join(item['hashtags'])}`\n\n"
        md_content += "---\n\n"

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n✅ Terminé ! Résultats enregistrés dans :")
    print(f"  - {args.output_md}")
    print(f"  - {args.output_json}\n")

    # Affichage récapitulatif console
    print("=" * 80)
    print("RÉCAPITULATIF DES TITRES GÉNÉRÉS")
    print("=" * 80)
    for item in results:
        print(f"\nClip #{item['index']} : {item['url']}")
        print(f"  • Titre Twitch original : {item['original_title']}")
        print(f"  • Jeu                   : {item['game_name']}")
        print(f"  • Titre YouTube Shorts  : {item['generated_youtube_title']}")
        print(f"  • Titre Overlay (haut)  : {item['generated_overlay_title']}")
        if len(item['variations']) > 1:
            print(f"  • Autres variantes      : {' | '.join(item['variations'][1:])}")
        print(f"  • Hashtags              : {' '.join(item['hashtags'])}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
