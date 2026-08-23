# scripts/generate_metadata.py
"""
Génération de métadonnées YouTube enrichies :
1. Essai via API Groq (Llama 3.1, gratuite) → titre accrocheur + hashtags
2. Fallback heuristique si l'API échoue / n'est pas configurée
"""

import os
import re
from datetime import datetime
import locale


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"


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


def _heuristic_title(clip_data):
    """Fallback : titre accrocheur base sur des templates + mots-cles."""
    title_raw = clip_data.get("title", "Un moment epique")
    title_clean = "".join(
        c for c in title_raw if c.isalnum() or c.isspace() or c in "'-_!?.,"
    ).strip()
    streamer = clip_data.get("broadcaster_name", "")
    game = clip_data.get("game_name") or ""

    # Patterns emotionnels => emoji
    emoji_map = {
        r"\b(rire|mdr|lol|drole|humour|rigol)\w*": "😂",
        r"\b(clutch|incroyable|epique|insane|ouf|wtf|omg)\w*": "🔥",
        r"\b(fail|rate|nul|naze|catastroph)\w*": "💀",
        r"\b(tr(i|e)ste|pleur|emotion|touchant)\w*": "😢",
        r"\b(rage|enerve|tilt|rageux)\w*": "😡",
        r"\b(beaux?|magnifique|style|propre|satisf[ai])\w*": "✨",
    }

    selected_emoji = "🎮"
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


def generate_youtube_metadata(clip_data):
    """
    Genere le bloc metadata pour l'upload YouTube Short.

    Returns:
        dict: {title, description, tags, categoryId, ...}
    """
    print("Generation des metadonnees...")

    titre_brut = clip_data.get("title", "Titre du clip")
    streamer = clip_data.get("broadcaster_name") or "Un streamer"
    game = clip_data.get("game_name") or "Gaming"
    clip_url = clip_data.get("url") or ""

    # --- Titre + hashtags ---
    if GROQ_API_KEY:
        title, htags = _groq_title_hashtags(clip_data)
    else:
        title, htags = None, None
        print("GROQ_API_KEY non defini => fallback heuristique.")

    if title is None:
        title = _heuristic_title(clip_data)
    if not htags:
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

    description = (
        f"🔻 DEROULE LA DESCRIPTION 🔻\n\n"
        f"📅 {today}\n"
        f"🎮 {game}\n"
        f"👤 @{streamer}\n\n"
        f"Montage : automatique 🤖\n\n"
        f"Chaine principale : @anyme0233\n\n"
        f"🔻 RESEAUX SOCIAUX 🔻\n\n"
        f"> Twitch | anyme023\n"
        f"> Instagram | anyme023\n"
        f"> Twitter | Anyme023Off\n"
        f"> TikTok | anyme023\n\n"
        f"ABONNE-TOI ! 🔔\n\n"
        f"Lien du clip originel << {titre_brut} >>:\n{clip_url}\n"
    )

    return {
        "title": title,
        "description": description,
        "tags": htags,
        "categoryId": "20",
        "privacyStatus": "public",
        "selfDeclaredMadeForKids": False,
        "embeddable": True,
        "license": "youtube",
    }