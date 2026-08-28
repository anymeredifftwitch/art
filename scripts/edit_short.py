# scripts/edit_short.py
"""
Montage vidéo unifié pour Shorts YouTube (9:16).

Pipeline :
1.  Hook d'intro (teaser 2-3s les plus intenses)
2.  Gameplay fullscreen 9:16 avec Ken Burns
3.  Sous-titres style TikTok (blanc sur fond pill semi-transparent)
4.  Flashs sur pics audio
5.  Barre de progression fine
6.  Titre overlay généré par IA en haut
7.  Séquence de fin
"""

import os
import tempfile
import numpy as np
from moviepy.editor import (
    VideoFileClip,
    CompositeVideoClip,
    AudioFileClip,
    CompositeAudioClip,
    TextClip,
    ColorClip,
    VideoClip,
    ImageClip,
    concatenate_videoclips,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
RESOLUTION = (1080, 1920)  # 9:16
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

FONT_BOLD = os.path.join(ASSETS_DIR, "OpenSans-Bold.ttf")
FONT_REGULAR = os.path.join(ASSETS_DIR, "OpenSans-Regular.ttf")
END_VIDEO = os.path.join(ASSETS_DIR, "fin_de_short.mp4")

# Vérification des polices
for p in [FONT_BOLD, FONT_REGULAR]:
    if not os.path.exists(p):
        print(f"⚠️  Police introuvable : {p} → fallback système")
        FONT_BOLD = "Arial-Bold"
        FONT_REGULAR = "Arial"
        break


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _text(text, font=FONT_BOLD, size=70, color="white",
          stroke_color="black", stroke_width=1.5):
    """TextClip sécurisé (fallback si police absente)."""
    kwargs = dict(
        txt=text,
        fontsize=size,
        color=color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
    )
    try:
        return TextClip(font=font, **kwargs)
    except Exception:
        return TextClip(**kwargs)


def _render_karaoke_subtitle_image(
    words_list,
    active_idx=-1,
    font_path=FONT_BOLD,
    font_size=52,
    active_color=(255, 230, 0, 255),    # Jaune fluo #FFE600
    inactive_color=(255, 255, 255, 255), # Blanc pur
    bg_color=(0, 0, 0, 160),             # Pilule noire semi-transparente
    padding_x=32,
    padding_y=16,
    radius=18,
):
    """
    Rend une image PIL représentant un groupe de sous-titres avec le mot
    actif en surbrillance jaune fluo et les autres en blanc.
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    word_texts = [
        w["word"].upper() if isinstance(w, dict) else str(w).upper()
        for w in words_list
    ]
    if not word_texts:
        return None

    space_w = font.getbbox(" ")[2] - font.getbbox(" ")[0]
    word_widths = []
    for wt in word_texts:
        bbox = font.getbbox(wt)
        word_widths.append(bbox[2] - bbox[0])

    total_words_w = sum(word_widths) + (len(word_texts) - 1) * space_w
    pill_w = max(240, min(1000, total_words_w + 2 * padding_x))
    pill_h = font_size + 2 * padding_y + 4

    img = Image.new("RGBA", (pill_w, pill_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Fond pilule arrondi
    draw.rounded_rectangle(
        [(0, 0), (pill_w, pill_h)],
        radius=radius,
        fill=bg_color,
    )

    cur_x = (pill_w - total_words_w) // 2
    for i, wt in enumerate(word_texts):
        color = active_color if i == active_idx else inactive_color
        bbox = font.getbbox(wt)
        offset_y = bbox[1]
        draw.text((cur_x, padding_y - offset_y), wt, font=font, fill=color)
        cur_x += word_widths[i] + space_w

    return img


def _generate_karaoke_subtitle_clips(group, full_dur):
    """
    Génère les clips de sous-titres karaoké pour un groupe de mots.
    Chaque mot prononcé est mis en surbrillance jaune en temps réel.
    """
    g_start = group.get("start", 0.0)
    g_end = group.get("end", 0.0)
    words = group.get("words", [])

    if g_end <= g_start or g_start >= full_dur:
        return []

    g_end = min(g_end, full_dur)

    # Si pas de liste de mots détaillée, fallback affichage blanc global
    if not words:
        raw_text = group.get("text", "")
        if not raw_text.strip():
            return []
        img = _render_karaoke_subtitle_image([raw_text], active_idx=-1)
        if img is None:
            return []
        arr = np.array(img)
        dur = g_end - g_start
        clip = ImageClip(arr[:, :, :3]).set_duration(dur)
        mask = ImageClip(arr[:, :, 3] / 255.0, ismask=True).set_duration(dur)
        return [clip.set_mask(mask).set_start(g_start).set_position(("center", 1150))]

    clips = []
    for i, w in enumerate(words):
        w_start = max(g_start, w.get("start", g_start))
        w_end = min(g_end, w.get("end", g_end))
        if w_end <= w_start:
            w_end = w_start + (g_end - g_start) / max(len(words), 1)

        dur = w_end - w_start
        if dur < 0.04 or w_start >= full_dur:
            continue

        img = _render_karaoke_subtitle_image(words, active_idx=i)
        if img is None:
            continue
        arr = np.array(img)
        clip = ImageClip(arr[:, :, :3]).set_duration(dur)
        mask = ImageClip(arr[:, :, 3] / 255.0, ismask=True).set_duration(dur)
        clips.append(clip.set_mask(mask).set_start(w_start).set_position(("center", 1150)))

    return clips


def _progress_bar(duration, target_w):
    """Barre de progression fine (hauteur 4px) en bas."""

    def make_frame(t):
        frame = np.zeros((4, target_w, 3), dtype=np.uint8)
        progress = min(t / max(duration, 0.01), 1.0)
        filled = int(progress * target_w)
        # Blanc semi-transparent
        frame[:, :filled] = [200, 200, 200]
        return frame

    return VideoClip(make_frame, duration=duration)


def _create_background(duration):
    """Fond sombre uni (plus propre que l'ancien fond théâtre)."""
    return ColorClip(RESOLUTION, color=(12, 12, 12)).set_duration(duration)


from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip


def _create_title_badge_image(
    text,
    max_width=920,
    font_path=FONT_BOLD,
    font_size=42,
    bg_color=(255, 255, 255, 255),
    text_color=(15, 15, 15, 255),
    radius=26,
    padding_x=40,
    padding_y=24,
    line_spacing=10,
):
    """
    Génère l'encadré blanc à coins arrondis (badge TikTok) avec word-wrap
    automatique et texte noir gras sans-serif centré. Ne coupe JAMAIS le texte.
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    words = text.strip().split()
    if not words:
        words = ["TITRE"]

    usable_width = max_width - (2 * padding_x)

    # Word wrap automatique
    lines = []
    current_line = []

    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = font.getbbox(test_line)
        line_w = bbox[2] - bbox[0]

        if line_w <= usable_width or not current_line:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    # Mesures de chaque ligne
    line_widths = []
    line_heights = []
    for line in lines:
        bbox = font.getbbox(line)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(font_size)

    max_line_w = max(line_widths) if line_widths else 100
    total_text_h = sum(line_heights) + (len(lines) - 1) * line_spacing

    badge_w = min(max_width, max(380, max_line_w + 2 * padding_x))
    badge_h = total_text_h + 2 * padding_y

    img = Image.new("RGBA", (badge_w, badge_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Fond blanc arrondi (badge TikTok)
    draw.rounded_rectangle(
        [(0, 0), (badge_w, badge_h)],
        radius=radius,
        fill=bg_color,
    )

    # Rendu des lignes de texte centrées
    current_y = padding_y
    for i, line in enumerate(lines):
        line_w = line_widths[i]
        line_x = (badge_w - line_w) // 2
        bbox = font.getbbox(line)
        offset_y = bbox[1]
        draw.text((line_x, current_y - offset_y), line, font=font, fill=text_color)
        current_y += line_heights[i] + line_spacing

    return img


def _title_badge_clip(text, duration):
    """
    Crée un clip MoviePy transparent avec le badge titre blanc arrondi,
    positionné de manière fixe en haut du frame.
    """
    if not text or not text.strip():
        return None

    img = _create_title_badge_image(text)
    arr = np.array(img)

    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3] / 255.0

    clip = ImageClip(rgb).set_duration(duration)
    mask = ImageClip(alpha, ismask=True).set_duration(duration)
    clip = clip.set_mask(mask)

    # Position fixe en haut du frame : centré horizontalement, marge haute Y=100px
    return clip.set_position(("center", 100))


def _generate_tts(text, max_duration):
    """
    Génère un fichier audio TTS (Google TTS) pour le texte donné.
    Renvoie un AudioFileClip calé sur max_duration.
    """
    if not text.strip():
        return None
    try:
        from gtts import gTTS
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        tts = gTTS(text=text, lang="fr", slow=False)
        tts.save(tmp.name)
        audio = AudioFileClip(tmp.name)
        # Si le TTS est plus court que le hook, on laisse le silence naturel
        if audio.duration > max_duration:
            audio = audio.subclip(0, max_duration)
        # Cleanup différé (on ne peut pas supprimer tant que moviepy lit)
        def _cleanup():
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
        return audio
    except Exception as e:
        print(f"⚠️  TTS indisponible (gTTS) : {e}")
        return None


def _fullscreen_zoom(clip):
    """
    Zoom centré pour remplir 1080x1920 sans letterboxing.
    Redimensionne pour couvrir les deux dimensions, puis crop le surplus.
    """
    # Échelle pour couvrir toute la surface (max des deux ratios)
    scale = max(RESOLUTION[0] / clip.w, RESOLUTION[1] / clip.h)
    new_w = int(clip.w * scale)
    new_h = int(clip.h * scale)
    c = clip.resize((new_w, new_h))
    c = c.crop(
        x_center=c.w / 2,
        width=RESOLUTION[0],
        y_center=c.h / 2,
        height=RESOLUTION[1],
    )
    if clip.audio is not None:
        c = c.set_audio(clip.audio)
    return c.set_position((0, 0))


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def edit_short(
    input_path,
    output_path,
    clip_data,
    webcam_info,
    subtitles,
    audio_analysis,
    max_duration=180,
):
    """
    Monte le clip complet.

    Args:
        input_path: chemin du clip MP4 brut
        output_path: chemin de sortie
        clip_data: {"title", "broadcaster_name", "game_name", ...}
        webcam_info: sortie de detect_webcam.detect_webcam()
        subtitles: sortie de generate_subtitles.transcribe()
        audio_analysis: sortie de analyze_audio.analyze_audio()
    """
    print("Début du montage unifié...")

    clip_raw = VideoFileClip(input_path)
    if clip_raw.duration > max_duration:
        clip_raw = clip_raw.subclip(0, max_duration)
    full_dur = clip_raw.duration
    peaks = audio_analysis.get("peaks", [])

    # ===================================================================
    # 1. Hook d'intro (teaser)
    # ===================================================================
    hook_start = audio_analysis.get("hook_start", 0.0)
    hook_end = min(audio_analysis.get("hook_end", 3.0), full_dur)
    hook = clip_raw.subclip(hook_start, hook_end)

    # Audio SFX Whoosh au tout début du clip (0s)
    whoosh_path = os.path.join(ASSETS_DIR, "sfx_whoosh.wav")
    if os.path.exists(whoosh_path) and hook.audio is not None:
        try:
            whoosh = AudioFileClip(whoosh_path).volumex(0.60)
            if whoosh.duration > hook.duration:
                whoosh = whoosh.subclip(0, hook.duration)
            hook_audio = CompositeAudioClip([hook.audio, whoosh.set_start(0.0)])
            hook = hook.set_audio(hook_audio)
        except Exception as e:
            print(f"⚠️  Erreur SFX hook : {e}")

    # Hook title: reuse the AI-generated overlay_title as hook text
    hook_title = clip_data.get("overlay_title", "")
    hook_title = "".join(c for c in hook_title if c.isprintable()).strip()
    if not hook_title or len(hook_title) < 3:
        hook_title = "BEST MOMENT"

    # Badge titre blanc arrondi (position fixe en haut dès 0s)
    hook_badge = _title_badge_clip(hook_title, hook.duration)

    hook_full = _fullscreen_zoom(hook)
    hook_elements = [hook_full]
    if hook_badge is not None:
        hook_elements.append(hook_badge)

    hook_comp = CompositeVideoClip(
        hook_elements,
        size=RESOLUTION,
    ).set_duration(hook.duration)

    # ===================================================================
    # 2. Éléments statiques du corps principal
    # ===================================================================
    bg = _create_background(full_dur)

    # Titre overlay : même badge blanc en haut pour toute la vidéo
    overlay_title = clip_data.get("overlay_title", "")
    overlay_title = "".join(c for c in overlay_title if c.isprintable()).strip()

    title_badge = None
    if overlay_title:
        title_badge = _title_badge_clip(overlay_title, full_dur)

    # ===================================================================
    # 3. Layout vidéo : Split-Screen (gameplay) ou Fullscreen Zoom (chatting)
    # ===================================================================
    clip_type = clip_data.get("clip_type", "gameplay")
    has_webcam = webcam_info.get("has_webcam", False)
    bbox = webcam_info.get("bbox")

    layout_elements = []

    if clip_type == "gameplay" and has_webcam and bbox:
        # Mode SPLIT-SCREEN :
        # - Zone Haute (1080 x 720) : Caméra du streamer agrandie
        # - Ligne de séparation (1080 x 4)
        # - Zone Basse (1080 x 1200) : Gameplay centré sur l'action
        bx1, by1, bx2, by2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2
        bw = max(bx2 - bx1, 1)

        # Cadrage confortable autour du visage/buste
        crop_w = max(bw * 1.8, 360)
        crop_h = crop_w * (720 / 1080)
        cx1 = max(0, int(cx - crop_w / 2))
        cx2 = min(clip_raw.w, int(cx + crop_w / 2))
        cy1 = max(0, int(cy - crop_h / 2))
        cy2 = min(clip_raw.h, int(cy + crop_h / 2))

        top_cam = (
            clip_raw.crop(x1=cx1, y1=cy1, x2=cx2, y2=cy2)
            .resize((RESOLUTION[0], 720))
            .set_duration(full_dur)
            .set_position((0, 0))
        )
        separator = (
            ColorClip((RESOLUTION[0], 4), color=(20, 20, 20))
            .set_duration(full_dur)
            .set_position((0, 718))
        )

        # Gameplay zone en bas
        target_aspect = 1080 / 1200
        gw, gh = clip_raw.w, clip_raw.h
        gameplay_crop_w = int(gh * target_aspect)
        if gameplay_crop_w > gw:
            gameplay_crop_w = gw
        gx1 = (gw - gameplay_crop_w) // 2
        gx2 = gx1 + gameplay_crop_w

        bottom_game = (
            clip_raw.crop(x1=gx1, y1=0, x2=gx2, y2=gh)
            .resize((RESOLUTION[0], 1200))
            .set_duration(full_dur)
            .set_position((0, 720))
        )
        layout_elements.extend([top_cam, bottom_game, separator])

    else:
        # Mode FULLSCREEN ZOOM centré (Just Chatting ou sans webcam)
        gameplay_zone = _fullscreen_zoom(clip_raw)
        KB_ZOOM_AMOUNT = 0.03

        def _kb_pos(t):
            zoom = 1.0 + KB_ZOOM_AMOUNT * (t / max(full_dur, 0.01))
            offset_x = (zoom - 1.0) * RESOLUTION[0] / 2
            offset_y = (zoom - 1.0) * RESOLUTION[1] / 2
            return (-offset_x, -offset_y)

        gameplay_clip = gameplay_zone.resize(
            (int(gameplay_zone.w * (1.0 + KB_ZOOM_AMOUNT)),
             int(gameplay_zone.h * (1.0 + KB_ZOOM_AMOUNT)))
        )
        gameplay_clip = gameplay_clip.set_position(
            lambda t: _kb_pos(t)
        ).set_duration(full_dur)
        layout_elements.append(gameplay_clip)

        # PIP webcam en haut à gauche si présent en Just Chatting
        if has_webcam and bbox:
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            pip = clip_raw.crop(x1=x1, y1=y1, x2=x2, y2=y2)
            pip_w = 220
            pip_h = int(pip_w * (y2 - y1) / max(x2 - x1, 1))
            pip = pip.resize((pip_w, pip_h)).set_duration(full_dur).set_position((20, 170))
            pip_border = ColorClip((pip_w + 6, pip_h + 6), color=(255, 255, 255)).set_duration(full_dur).set_opacity(0.8).set_position((17, 167))
            layout_elements.extend([pip_border, pip])

    # Composition de base
    base_elements = [bg] + layout_elements
    if title_badge is not None:
        base_elements.append(title_badge)

    # ===================================================================
    # 4. Flashs sur pics audio
    # ===================================================================
    for peak_t in peaks:
        if peak_t >= full_dur:
            continue
        flash = ColorClip(RESOLUTION, color=(255, 255, 255))
        flash = flash.set_duration(0.08).set_start(peak_t).set_opacity(0.15)
        base_elements.append(flash)

    # ===================================================================
    # 5. Sous-titres Karaoké mot par mot (surbrillance jaune TikTok)
    # ===================================================================
    subtitle_layers = []
    if subtitles:
        for group in subtitles:
            clips = _generate_karaoke_subtitle_clips(group, full_dur)
            subtitle_layers.extend(clips)

    # ===================================================================
    # 6. Barre de progression (pas de CTA : déjà dans le clip Twitch)
    # ===================================================================
    prog = _progress_bar(full_dur, RESOLUTION[0]).set_position(("center", 1916))

    # ===================================================================
    # 7. Composition finale du corps
    # ===================================================================
    corp = CompositeVideoClip(
        base_elements + subtitle_layers + [prog],
        size=RESOLUTION,
    ).set_duration(full_dur)

    # Restaurer l'audio (perdu par les crops/resizes)
    if clip_raw.audio is not None:
        corp = corp.set_audio(clip_raw.audio)

    # ===================================================================
    # 8. Séquence de fin + concaténation
    # ===================================================================
    end_clips = [hook_comp, corp]

    if os.path.exists(END_VIDEO):
        try:
            outro = VideoFileClip(END_VIDEO).resize(RESOLUTION)
            if outro.duration > 1.5:
                outro = outro.subclip(0, 1.5)
            end_clips.append(outro)
        except Exception as e:
            print(f"⚠️  Outro ignorée ({e})")

    final = concatenate_videoclips(end_clips, method="compose")

    # ===================================================================
    # 9. Écriture
    # ===================================================================
    print(f"Rendu final vers {output_path} ...")
    final.write_videofile(
        output_path,
        fps=30,
        codec="libx264",
        audio_codec="aac",
    )

    # Fermeture propre
    clip_raw.close()
    hook.close()
    corp.close()
    final.close()

    print(f"Montage terminé : {output_path}")