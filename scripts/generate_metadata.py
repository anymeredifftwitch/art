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


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-70b-versatile"

# Noms de streamer à supprimer des titres (insensible à la casse)
_STREAMER_BLACKLIST = [
    "anyme023", "anyme", "anyme0233", "anymeoff",
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
        title = _groq_video_title(game, transcription_text)
        if title:
            return title

    # Fallback heuristique
    return _heuristic_video_title(transcription_text,
                                  clip_data.get("title", ""))


def _groq_video_title(game, transcription):
    """Appelle Groq pour un titre overlay ultra-court et viral."""
    prompt = (
        "Tu es un expert en titres viraux pour Shorts/TikTok. "
        "À partir de la transcription audio d'un clip Twitch, "
        "crée UN SEUL titre accrocheur à afficher EN HAUT de la vidéo.\n\n"
        "Règles impératives :\n"
        "- Style POV / storytelling / intrigue / drama / punchline\n"
        "- Résume l'idée la plus virale du clip en une phrase choc\n"
        "- Tu peux dramatiser ou inventer un contexte pour rendre ça viral\n"
        "- MAXIMUM 38 caractères (sinon ça déborde à l'écran)\n"
        "- TOUT EN MAJUSCULES\n"
        "- AUCUN emoji, AUCUN hashtag, AUCUN nom de streamer\n"
        "- Réponds UNIQUEMENT le titre, rien d'autre\n\n"
        f"Jeu : {game or 'Inconnu'}\n"
        f"Transcription : << {transcription[:400]} >>"
    )

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Tu réponds uniquement le titre, sans guillemets ni ponctuation superflue."},
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
            max_tokens=40,
        )
        title = resp.choices[0].message.content.strip().upper()
        # Nettoyage
        title = title.strip('"').strip("'").strip()
        title = _clean_title(title)
        if len(title) > 42:
            title = title[:39].strip() + "..."
        if not title:
            return None
        print(f"🎬 Titre overlay Groq : {title}")
        return title

    except Exception as exc:
        print(f"⚠️  Groq titre overlay indisponible ({exc}), fallback.")
        return None


def _heuristic_video_title(transcription, clip_title_raw):
    """
    Fallback : génère un titre à partir de la transcription
    avec des templates viraux.
    """
    text = transcription.lower() if transcription else clip_title_raw.lower()

    # Templates par mot-clé dans la transcription
    templates = [
        # (mots-clés, titre généré)
        (["tromp", "ment", "trahi", "cach"], "POV: IL M'A TROMPÉ EN LIVE"),
        (["peur", "flipp", "horreur", "jumpscare"], "LA PEUR DE SA VIE"),
        (["pleur", "triste", "emotion"], "IL FOND EN LARMES"),
        (["rage", "énerve", "tilt", "casse"], "IL PÈTE UN CÂBLE"),
        (["rire", "mdr", "lol", "hilar"], "MORT DE RIRE EN DIRECT"),
        (["clutch", "incroyable", "ouf", "wtf", "omg"], "UNE CLUTCH DE MALADE"),
        (["fail", "raté", "nul"], "FAIL ÉPIQUE EN LIVE"),
        (["gagn", "victoir", "win"], "IL A ENFIN GAGNÉ"),
        (["perd", "defaite", "mort"], "DÉTRUIT EN DIRECT"),
        (["chanter", "chanson", "musique"], "IL CHANTE EN LIVE"),
        (["danse", "danser"], "IL SE LÂCHE SUR LE LIVE"),
    ]

    for keywords, title in templates:
        if any(kw in text for kw in keywords):
            return title[:42]

    # Fallback : extrait intelligent des 6-8 premiers mots pertinents
    if transcription and len(transcription) > 5:
        # Enlever les mots parasites (euh, bah, ouais, etc.)
        filler = {"euh", "bah", "ben", "hein", "quoi", "du coup", "genre", "voilà",
                   "oui", "ouais", "non", "yes", "ok", "okay"}
        words = [w for w in transcription.split()
                 if w.lower() not in filler and len(w) > 2][:6]
        if words:
            snippet = " ".join(words).upper()
            if len(snippet) > 32:
                snippet = snippet[:29].strip() + "..."
            return snippet[:42]

    # Dernier recours
    return "ANYME EN LIVE"[:42]


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

    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
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

        print(f"Groq => titre: {title}")
        print(f"Groq => hashtags: {hashtags}")
        return title, hashtags

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