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
    streamer = (clip_data.get("broadcaster_name") or "ANYME").upper()
    game = clip_data.get("game_name") or ""

    # Construire le texte de la transcription
    transcription_text = ""
    if subtitles:
        transcription_text = " ".join(
            s.get("text", "") for s in subtitles if s.get("text")
        ).strip()

    # Essayer Groq si configuré et si on a de la matière
    if GROQ_API_KEY and transcription_text:
        title = _groq_video_title(streamer, game, transcription_text)
        if title:
            return title

    # Fallback heuristique
    return _heuristic_video_title(streamer, transcription_text,
                                  clip_data.get("title", ""))


def _groq_video_title(streamer, game, transcription):
    """Appelle Groq pour un titre overlay ultra-court et viral."""
    prompt = (
        "Tu es un expert en titres viraux pour Shorts/TikTok. "
        "À partir de la transcription audio d'un clip Twitch, "
        "crée UN SEUL titre à afficher EN HAUT de la vidéo.\n\n"
        "Règles impératives :\n"
        "- Style POV / storytelling / intrigue / drama\n"
        "- Tu peux dramatiser ou inventer un contexte pour le rendre viral\n"
        f"- Inclus le nom du streamer : {streamer}\n"
        "- MAXIMUM 38 caractères (sinon ça déborde)\n"
        "- TOUT EN MAJUSCULES\n"
        "- Aucun emoji, aucun hashtag\n"
        "- Réponds UNIQUEMENT le titre, rien d'autre\n\n"
        f"Streamer : {streamer}\n"
        f"Jeu : {game or 'Inconnu'}\n"
        f"Transcription : << {transcription[:300]} >>"
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
        if len(title) > 42:
            title = title[:39].strip() + "..."
        print(f"🎬 Titre overlay Groq : {title}")
        return title

    except Exception as exc:
        print(f"⚠️  Groq titre overlay indisponible ({exc}), fallback.")
        return None


def _heuristic_video_title(streamer, transcription, clip_title_raw):
    """
    Fallback : génère un titre à partir de la transcription
    avec des templates viraux.
    """
    text = transcription.lower() if transcription else clip_title_raw.lower()

    # Templates par mot-clé dans la transcription
    templates = [
        # (mots-clés, titre généré)
        (["tromp", "ment", "trahi", "cach"], f"POV: {streamer} SE FAIT TROMPER"),
        (["peur", "flipp", "horreur", "jumpscare"], f"{streamer} A EU LA PEUR DE SA VIE"),
        (["pleur", "triste", "emotion"], f"{streamer} FOND EN LARMES"),
        (["rage", "énerve", "tilt", "casse"], f"{streamer} PÈTE UN CABLE"),
        (["rire", "mdr", "lol", "hilar"], f"{streamer} MORT DE RIRE"),
        (["clutch", "incroyable", "ouf", "wtf", "omg"], f"{streamer} CLUTCH DE MALADE"),
        (["fail", "raté", "nul"], f"{streamer} FAIL ÉPIQUE"),
        (["gagn", "victoir", "win"], f"{streamer} GAGNE ENFIN"),
        (["perd", "defaite", "mort"], f"{streamer} DÉTRUIT EN DIRECT"),
        (["chanter", "chanson", "musique"], f"{streamer} CHANTE EN LIVE"),
        (["danse", "danser"], f"{streamer} SE LACHE SUR LE DANCEFLOOR"),
    ]

    for keywords, title in templates:
        if any(kw in text for kw in keywords):
            return title[:42]

    # Fallback générique basé sur les premiers mots
    if transcription and len(transcription) > 5:
        words = transcription.split()[:6]
        snippet = " ".join(words).upper()
        if len(snippet) > 30:
            snippet = snippet[:27].strip() + "..."
        return f"{streamer}: {snippet}"[:42]

    # Dernier recours
    return f"{streamer} EN LIVE"[:42]


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
        "TITRE: <titre accrocheur max 95 caracteres avec emojis>\n"
        "HASHTAGS: #tag1 #tag2 #tag3 #tag4 #tag5\n\n"
        f"Contexte :\n- Streamer : {streamer}\n- Jeu : {game}\n"
        f"- Titre du clip original : << {title_raw} >>\n\n"
        "Regles :\n"
        "- Titre punchy, style TikTok/Shorts, avec 1-2 emojis\n"
        "- Hashtags : 1 pour le jeu, 1 pour le streamer, 1 pour 'TwitchFR', "
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
        hashtags_raw = hashtags_match.group(1).strip() if hashtags_match else ""
        hashtags = [t.strip() for t in hashtags_raw.split() if t.startswith("#")]

        print(f"Groq => titre: {title}")
        print(f"Groq => hashtags: {hashtags}")
        return title, hashtags

    except Exception as exc:
        print(f"API Groq indisponible ({exc}), fallback heuristique.")
        return None, None


def _pick_emoji(text):
    """Choisit un emoji en fonction du contenu du texte."""
    t = text.lower()
    if any(w in t for w in ("tromp", "ment", "trahi")): return "😱"
    if any(w in t for w in ("peur", "flipp", "horreur")): return "😨"
    if any(w in t for w in ("pleur", "triste", "larme")): return "😢"
    if any(w in t for w in ("rage", "pète", "cable")): return "🤬"
    if any(w in t for w in ("rire", "mort de rire", "mdr")): return "😂"
    if any(w in t for w in ("clutch", "incroyable", "malade")): return "🔥"
    if any(w in t for w in ("fail", "raté")): return "💀"
    if any(w in t for w in ("gagn", "victoir", "win")): return "🏆"
    return "🎮"


def _heuristic_title(clip_data):
    """Fallback : titre accrocheur base sur des templates + mots-cles."""
    title_raw = clip_data.get("title", "Un moment epique")
    title_clean = "".join(
        c for c in title_raw if c.isalnum() or c.isspace() or c in "'-_!?.,"
    ).strip()
    streamer = clip_data.get("broadcaster_name", "")
    game = clip_data.get("game_name") or ""

    emoji_map = {
        r"\b(rire|mdr|lol|drole|humour|rigol)\w*": "\U0001f602",
        r"\b(clutch|incroyable|epique|insane|ouf|wtf|omg)\w*": "\U0001f525",
        r"\b(fail|rate|nul|naze|catastroph)\w*": "\U0001f480",
        r"\b(tr(i|e)ste|pleur|emotion|touchant)\w*": "\U0001f622",
        r"\b(rage|enerve|tilt|rageux)\w*": "\U0001f620",
        r"\b(beaux?|magnifique|style|propre|satisf[ai])\w*": "\u2728",
    }

    selected_emoji = "\U0001f3ae"
    for pattern, emoji in emoji_map.items():
        if re.search(pattern, title_clean.lower()):
            selected_emoji = emoji
            break

    title = f"{selected_emoji} {title_clean}"
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
    # (en minuscules pour le SEO YouTube, avec emojis en plus)
    if video_title:
        # Ajouter un emoji devant pour le rendre plus cliquable sur YouTube
        emoji = _pick_emoji(video_title)
        title = f"{emoji} {video_title.capitalize()}"[:100]
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