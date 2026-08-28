# scripts/generate_metadata.py
"""
Generation de metadonnees YouTube enrichies :
1. Essai via API Groq (gratuite) -> titre accrocheur + hashtags
2. Fallback heuristique si l'API echoue / n'est pas configuree
"""

import os
import re
from datetime import datetime
import locale


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Noms de comptes/handles spécifiques à supprimer des titres (insensible à la casse)
_STREAMER_BLACKLIST = [
    "anyme023", "anyme0233", "anymeoff",
]


def _clean_title(title):
    """
    Nettoie un titre généré par IA : supprime les emojis, noms de streamer,
    hashtags, et artefacts résiduels.
    """
    import unicodedata

    # 1. Supprimer les emojis (Unicode ranges + séquences)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols extended-A
        "\U00002600-\U000026FF"  # misc symbols
        "\U0000FE00-\U0000FE0F"  # variation selectors
        "\U0000200D"             # zero width joiner
        "\U0000200C"             # zero width non-joiner
        "]+",
        flags=re.UNICODE,
    )
    title = emoji_pattern.sub("", title)

    # 2. Supprimer les noms de streamer (insensible casse)
    for name in _STREAMER_BLACKLIST:
        # Match avec prefix/suffix non-alpha pour éviter de tronquer des mots normaux
        title = re.sub(
            rf"\b{re.escape(name)}\b",
            "",
            title,
            flags=re.IGNORECASE,
        )

    # 3. Supprimer les hashtags résiduels
    title = re.sub(r"#\w+", "", title)

    # 4. Nettoyage final : espaces multiples, ponctuation orpheline
    title = re.sub(r"\s+", " ", title)
    title = title.strip(" -|:;,.!?·")
    title = title.strip()

    return title

# ---------------------------------------------------------------------------
# Titre overlay vidéo (basé sur transcription audio)
# ---------------------------------------------------------------------------

def generate_video_title(clip_data, subtitles=None):
    """
    Génère un titre clickbait pour l'overlay vidéo, basé sur
    la transcription audio du clip.

    Returns:
        str : titre court (max 40 car.) en majuscules, sans emoji
    """
    game = clip_data.get("game_name") or ""

    # Construire le texte de la transcription
    transcription_text = ""
    if subtitles:
        transcription_text = " ".join(
            s.get("text", "") for s in subtitles if s.get("text")
        ).strip()

    # Essayer Groq si configuré et si on a de la matière
    if GROQ_API_KEY and transcription_text:
        title = _groq_video_title(game, transcription_text, clip_title_raw=clip_data.get("title", ""))
        if title:
            return title

    # Fallback heuristique
    return _heuristic_video_title(transcription_text,
                                  clip_data.get("title", ""))


def _groq_video_title(game, transcription, clip_title_raw=""):
    """Appelle Groq pour un titre overlay percutant et viral adapté à l'audience Twitch FR."""
    prompt = (
        "Tu es un expert d'élite en création de titres viraux pour Shorts YouTube et TikTok (audience Twitch FR / Gaming).\n"
        "À partir du contexte et de la transcription audio d'un clip Twitch du streamer Anyme, crée UN SEUL titre overlay ultra-percutant à afficher en haut de la vidéo.\n\n"
        "STYLE & TON EXIGÉS :\n"
        "- Langage direct, familier, drôle, sans filtre, style commu Twitch FR / TikTok\n"
        "- Formats types autorisés :\n"
        "  1. 'POV: Anyme [action choc] en LIVE' (ex: POV: Anyme pète son crâne en LIVE)\n"
        "  2. 'POV: IL [action choc]' (ex: POV: IL SE CHIE DESSUS DE PEUR)\n"
        "  3. Phrase choc en MAJUSCULES (ex: IL A DIT QUOI LA ??? / LES VIEWERS SONT FOUS)\n"
        "  4. 'Anyme [exploit/fail] et [punchline]' (ex: Anyme CLUTCH et se PISSE DESSUS / ANYME CLAQUE UN SMIC SUR CS2)\n"
        "- Longueur : 20 à 42 caractères MAXIMUM (court pour tenir à l'écran)\n"
        "- Emphase en MAJUSCULES sur les mots clés ou tout en majuscules\n"
        "- AUCUN emoji, AUCUN hashtag\n"
        "- Réponds UNIQUEMENT le titre, rien d'autre.\n\n"
        "EXEMPLES DE TITRES PARFAITS :\n"
        "- POV: Anyme pète son crâne en LIVE\n"
        "- POV: Anyme raconte sa PIRE ANECDOTE en LIVE\n"
        "- IL A DIT QUOI LA ???\n"
        "- POV: Anyme FOU LE FEU EN LIVE\n"
        "- LES VIEWERS SONT FOUS\n"
        "- Anyme CLUTCH et se PISSE DESSUS\n"
        "- POV: IL SE CHIE DESSUS DE PEUR\n"
        "- ANYME CLAQUE UN SMIC SUR CS2\n"
        "- POV ANYME DRAGUE UN TRANS EN LIVE\n"
        "- POV: ANYME SE FAIL et PLEURE\n\n"
        f"Jeu : {game or 'Just Chatting'}\n"
        f"Titre Twitch original : << {clip_title_raw} >>\n"
        f"Transcription : << {transcription[:400]} >>"
    )

    candidate_models = [GROQ_MODEL, "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]
    models_to_try = list(dict.fromkeys(candidate_models))

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        for m in models_to_try:
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": "Tu réponds uniquement le titre overlay court, sans guillemets ni blabla."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.9,
                    max_tokens=80,
                )
                title = resp.choices[0].message.content.strip()
                title = title.strip('"').strip("'").strip()
                title = _clean_title(title)
                if len(title) > 90:
                    title = title[:87].strip() + "..."
                if title:
                    print(f"🎬 Titre overlay Groq ({m}) : {title}")
                    return title
            except Exception as e:
                print(f"⚠️ Modèle Groq '{m}' indisponible ({e}), essai du modèle suivant...")
                continue

        return None

    except Exception as exc:
        print(f"⚠️ Groq titre overlay indisponible ({exc}), fallback.")
        return None


def _heuristic_video_title(transcription, clip_title_raw):
    """
    Fallback : génère un titre à partir de la transcription
    avec les templates viraux calibrés sur le style Twitch FR d'Anyme.
    """
    text = (transcription + " " + clip_title_raw).lower()

    # Templates par mot-clé avec frontières de mots pour éviter les faux-positifs
    templates = [
        ([r"police", r"flic", r"poursuite", r"prison"], "POV: Anyme FUIT LA POLICE EN LIVE"),
        ([r"pack opening", r"crédit", r"animation", r"poubelle", r"scam"], "POV: Anyme SE FAIT SCAM EN LIVE"),
        ([r"only up", r"montée", r"tuyau", r"désinstalle", r"tout en bas"], "POV: Anyme PÈTE SON CRÂNE SUR ONLY UP"),
        ([r"mariage", r"bague", r"poème", r"discord"], "UN VIEWER LE DEMANDE EN MARIAGE"),
        ([r"hardcore", r"creeper", r"supprimé", r"diamant"], "POV: IL PERD TOUT EN HARDCORE"),
        ([r"carapace", r"mario kart", r"éclair", r"vol du siècle"], "LE PIRE VOL SUR MARIO KART"),
        ([r"uber", r"livreur", r"burger", r"frite", r"sauce mayo"], "POV: LE LIVREUR UBER L'A ARNAQUÉ"),
        ([r"vaisseau", r"lethal", r"porte", r"laissé", r"crever"], "SES POTES L'ABANDONNENT EN LIVE"),
        ([r"beat saber", r"expert\+", r"mains", r"combo"], "Anyme DEVIENT UN MONSTRE EN LIVE"),
        ([r"australie", r"capitale", r"géographie", r"quiz", r"sydney"], "POV: IL S'AFFICHE SUR UN QUIZ"),
        ([r"peur", r"flipp", r"horreur", r"jumpscare", r"crise cardiaque"], "POV: IL SE CHIE DESSUS DE PEUR"),
        ([r"smic", r"prix", r"skin", r"2000", r"cher", r"steam", r"cs2", r"cs:go"], "ANYME CLAQUE UN SMIC SUR CS2"),
        ([r"clutch", r"one tap", r"l'as", r"\bace\b", r"patron", r"1v4", r"1v3", r"\bwin\b"], "Anyme CLUTCH et se PISSE DESSUS"),
        ([r"anecdote", r"gênant", r"honte", r"histoire", r"métro"], "POV: Anyme raconte sa PIRE ANECDOTE en LIVE"),
        ([r"feu", r"fumée", r"cramé", r"pc", r"setup", r"brûle"], "POV: Anyme FOU LE FEU EN LIVE"),
        ([r"chat", r"monstre", r"troll", r"viewers", r"déteste"], "LES VIEWERS SONT FOUS"),
        ([r"\bquoi\b", r"répète", r"malade mental", r"dit quoi"], "IL A DIT QUOI LA ???"),
        ([r"crush", r"drague", r"bégayé", r"\btrans\b"], "POV ANYME DRAGUE UN TRANS EN LIVE"),
        ([r"fail", r"raté", r"pad", r"vide", r"tombe", r"débutant", r"pleure"], "POV: ANYME SE FAIL et PLEURE"),
        ([r"rage", r"énerve", r"tilt", r"crâne", r"casse", r"pète", r"câble", r"complot", r"écoute pas"], "POV: Anyme pète son crâne en LIVE"),
    ]

    for patterns, title in templates:
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                return title

    # Fallback intelligent si mots spécifiques
    if transcription and len(transcription) > 5:
        return f"POV: Anyme en LIVE"

    return "POV: Anyme en LIVE"


# ---------------------------------------------------------------------------
# Métadonnées YouTube (titre SEO + description + hashtags)
# ---------------------------------------------------------------------------

def _groq_title_hashtags(clip_data):
    """Appelle l'API Groq pour un titre accrocheur + 5 hashtags."""

    title_raw = clip_data.get("title", "clip")
    streamer = clip_data.get("broadcaster_name", "streamer")
    game = clip_data.get("game_name") or "Gaming"

    prompt = (
        "Tu es un expert en marketing YouTube Shorts. "
        "Genere UNIQUEMENT ce format de reponse (pas de blabla) :\n\n"
        "TITRE: <titre accrocheur max 95 caracteres, SANS emoji>\n"
        "HASHTAGS: #tag1 #tag2 #tag3 #tag4 #tag5\n\n"
        f"Contexte :\n- Streamer : {streamer}\n- Jeu : {game}\n"
        f"- Titre du clip original : << {title_raw} >>\n\n"
        "Regles :\n"
        "- Titre punchy, style TikTok/Shorts, AUCUN emoji\n"
        "- Commence par 'Anyme | ' suivi du titre\n"
        "- Hashtags : 1 pour le jeu, 1 pour 'Anyme', 1 pour 'TwitchFR', "
        "2 pertinents au contenu du clip\n"
    )

    candidate_models = [GROQ_MODEL, "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]
    models_to_try = list(dict.fromkeys(candidate_models))

    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)

        for m in models_to_try:
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": "Tu reponds strictement dans le format demande."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.9,
                    max_tokens=200,
                )
                content = resp.choices[0].message.content.strip()

                title_match = re.search(r"TITRE:\s*(.+?)(?:\n|$)", content)
                hashtags_match = re.search(r"HASHTAGS:\s*(.+?)(?:\n|$)", content)

                title = (title_match.group(1).strip() if title_match else title_raw)[:100]
                title = _clean_title(title)
                hashtags_raw = hashtags_match.group(1).strip() if hashtags_match else ""
                hashtags = [t.strip() for t in hashtags_raw.split() if t.startswith("#")]

                if title:
                    print(f"Groq ({m}) => titre: {title}")
                    print(f"Groq ({m}) => hashtags: {hashtags}")
                    return title, hashtags
            except Exception as e:
                print(f"⚠️ Modèle Groq '{m}' indisponible ({e}), essai du modèle suivant...")
                continue

        return None, None

    except Exception as exc:
        print(f"API Groq indisponible ({exc}), fallback heuristique.")
        return None, None


def _heuristic_title(clip_data):
    """Fallback : titre accrocheur base sur le titre du clip (sans emoji)."""
    title_raw = clip_data.get("title", "Un moment epique")
    title_clean = "".join(
        c for c in title_raw if c.isalnum() or c.isspace() or c in "'-_!?.,"
    ).strip()
    game = clip_data.get("game_name") or ""

    title = f"Anyme | {title_clean}"
    if len(title) > 100:
        title = title[:97].strip() + "..."
    return title


def _heuristic_hashtags(clip_data):
    """Fallback : hashtags bases sur les donnees du clip."""
    tags = {"twitchfr", "shorts", "anyme023", "gaming"}

    game = (clip_data.get("game_name") or "").lower().strip()
    if game and game not in ("just chatting", ""):
        clean = game.replace(" ", "").replace(":", "").replace("'", "")
        tags.add(clean)

    title_raw = clip_data.get("title", "")
    important = re.findall(r"\b\w{4,}\b", title_raw.lower())
    for w in important[:4]:
        tags.add(w)

    return [f"#{t}" for t in sorted(tags)]


def generate_youtube_metadata(clip_data, video_title=None):
    """
    Genere le bloc metadata pour l'upload YouTube Short.

    Args:
        clip_data: données du clip Twitch
        video_title: titre déjà généré pour l'overlay (optionnel,
                     évite un 2e appel Groq)

    Returns:
        dict: {title, description, tags, categoryId, ...}
    """
    print("Generation des metadonnees...")

    streamer = clip_data.get("broadcaster_name") or "Un streamer"
    game = clip_data.get("game_name") or "Gaming"
    clip_url = clip_data.get("url") or ""

    # --- Titre ---
    # Si un titre overlay a déjà été généré, on le réutilise
    # (sans emoji, clean)
    if video_title:
        title = f"Anyme | {video_title.capitalize()}"[:100]
        print(f"Titre YouTube (depuis overlay) : {title}")
    elif GROQ_API_KEY:
        title, _ = _groq_title_hashtags(clip_data)
        if title is None:
            title = _heuristic_title(clip_data)
    else:
        print("GROQ_API_KEY non defini => fallback heuristique.")
        title = _heuristic_title(clip_data)

    # --- Hashtags (ne rappelle pas Groq si déjà fait ci-dessus) ---
    if video_title:
        # Pas d'appel Groq pour le titre, on en fait un pour les hashtags
        if GROQ_API_KEY:
            _, htags = _groq_title_hashtags(clip_data)
        else:
            htags = None
        if not htags:
            htags = _heuristic_hashtags(clip_data)
    elif GROQ_API_KEY:
        # Le titre vient déjà d'un appel Groq → ne pas en refaire un 2e
        htags = _heuristic_hashtags(clip_data)
    else:
        htags = _heuristic_hashtags(clip_data)

    # --- Description ---
    try:
        locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")
    except Exception:
        try:
            locale.setlocale(locale.LC_TIME, "fr_FR")
        except Exception:
            pass

    today = datetime.now().strftime("%d %B %Y")

    # IMPORTANT : pas de < > dans la description (interdits par l'API YouTube)
    description = (
        "DEROULE LA DESCRIPTION !\n\n"
        f"{today}\n"
        f"Jeu : {game}\n"
        f"Streamer : @{streamer}\n\n"
        "Montage : automatique\n\n"
        "Chaine principale : @anyme0233\n\n"
        "RESEAUX SOCIAUX :\n"
        "- Twitch : anyme023\n"
        "- Instagram : anyme023\n"
        "- Twitter : Anyme023Off\n"
        "- TikTok : anyme023\n\n"
        "ABONNE-TOI !\n\n"
        f"Clip original : {clip_url}\n"
    )

    return {
        "title": title[:100],
        "description": description,
        "tags": htags,
        "categoryId": "20",
        "privacyStatus": "public",
        "selfDeclaredMadeForKids": False,
        "embeddable": True,
        "license": "youtube",
    }