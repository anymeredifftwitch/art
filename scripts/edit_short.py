# scripts/edit_short.py
"""
Montage vidéo unifié pour Shorts YouTube (9:16).

Pipeline :
1.  Hook d'intro (teaser 2-3s les plus intenses)
2.  Gameplay fullscreen 9:16 avec Ken Burns
3.  Webcam en PIP (picture-in-picture) en haut à droite
4.  Sous-titres centrés (texte blanc avec contour, pas de fond)
5.  Flashs sur pics audio
6.  Barre de progression fine
7.  Call-to-action discret (bas de l'écran)
8.  Titre en haut sur fond dégradé
9.  Séquence de fin
"""

import os
import numpy as np
from moviepy.editor import (
    VideoFileClip,
    CompositeVideoClip,
    TextClip,
    ColorClip,
    VideoClip,
    concatenate_videoclips,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
RESOLUTION = (1080, 1920)  # 9:16
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

FONT_BOLD = os.path.join(ASSETS_DIR, "Roboto-Bold.ttf")
FONT_REGULAR = os.path.join(ASSETS_DIR, "Roboto-Regular.ttf")
END_VIDEO = os.path.join(ASSETS_DIR, "fin_de_short.mp4")

# Vérification des polices
for p in [FONT_BOLD, FONT_REGULAR]:
    if not os.path.exists(p):
        print(f"⚠️  Police introuvable : {p} → fallback système")
        FONT_BOLD = "Arial-Bold"
        FONT_REGULAR = "Arial"
        break

# PIP webcam
PIP_SIZE_RATIO = 0.22       # Largeur du PIP = 22% de l'écran
PIP_MARGIN = 16             # Marge depuis les bords
PIP_BORDER = 4              # Épaisseur de la bordure blanche


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


def _subtitle_clip(group, duration):
    """Sous-titre simple : texte blanc avec contour noir (pas de fond)."""
    text = group.get("text", "")
    if not text.strip():
        return None

    tc = _text(
        text,
        font=FONT_BOLD,
        size=52,
        color="white",
        stroke_color="black",
        stroke_width=3.5,
    )
    tc = tc.set_duration(duration)
    return tc


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


def _cta_clip(text, start_time, duration):
    """Call-to-action discret : fade in/out en bas."""

    def _opacity(t):
        local_t = t - start_time
        if local_t < 0.0:
            return 0.0
        fade_in = min(local_t / 0.2, 1.0)
        if local_t > 1.8:
            fade_out = max(0.0, (2.0 - local_t) / 0.2)
            return min(fade_in, fade_out)
        return fade_in

    tc = _text(
        text,
        font=FONT_BOLD,
        size=38,
        color="white",
        stroke_color="black",
        stroke_width=2.0,
    )
    tc = (
        tc.set_duration(duration)
        .set_position(("center", 1740))
        .set_opacity(_opacity)
    )
    return tc


def _create_background(duration):
    """Fond sombre uni (plus propre que l'ancien fond théâtre)."""
    return ColorClip(RESOLUTION, color=(12, 12, 12)).set_duration(duration)


def _top_gradient(duration):
    """Bandeau sombre en haut pour lisibilité du titre (dégradé)."""
    h = 200
    bar = ColorClip((RESOLUTION[0], h), color=(0, 0, 0)).set_duration(duration)

    # Masque dégradé : opaque en haut, transparent en bas
    def make_mask(t):
        mask = np.zeros((h, RESOLUTION[0]), dtype=np.uint8)
        for y in range(h):
            alpha = 1.0 - (y / h) ** 1.5
            mask[y, :] = int(255 * alpha)
        return mask

    mask_clip = VideoClip(make_mask, duration=duration, ismask=True)
    bar = bar.set_mask(mask_clip)
    return bar.set_position((0, 0))


def _fullscreen_zoom(clip):
    """Zoom centré pour remplir 1080x1920 (garde l'audio)."""
    c = clip.resize(height=RESOLUTION[1])
    c = c.crop(
        x_center=c.w / 2,
        width=RESOLUTION[0],
        y_center=c.h / 2,
        height=RESOLUTION[1],
    )
    if clip.audio is not None:
        c = c.set_audio(clip.audio)
    return c.set_position((0, 0))


def _pip_webcam(clip_raw, bbox, full_dur):
    """
    Crée un PIP webcam (rectangle avec bordure blanche) en haut à droite.
    Retourne le clip PIP prêt à être composité, ou None.
    """
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

    # Crop de la zone visage
    face = clip_raw.crop(x1=x1, y1=y1, x2=x2, y2=y2)

    # Redimensionner en PIP
    pip_w = int(RESOLUTION[0] * PIP_SIZE_RATIO)
    face = face.resize(width=pip_w)

    # Bordure blanche : rectangle blanc légèrement plus grand derrière
    border_w = pip_w + PIP_BORDER * 2
    border_h = face.h + PIP_BORDER * 2
    border = ColorClip((border_w, border_h), color=(255, 255, 255))
    border = border.set_duration(full_dur).set_opacity(0.85)

    # Position en haut à droite
    pip_x = RESOLUTION[0] - border_w - PIP_MARGIN
    pip_y = 160  # Sous le titre

    comp = CompositeVideoClip(
        [
            border.set_position((0, 0)),
            face.set_position((PIP_BORDER, PIP_BORDER)),
        ],
        size=(border_w, border_h),
    ).set_duration(full_dur).set_position((pip_x, pip_y))

    return comp


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

    hook_label = (
        _text("🔥 BEST MOMENT", font=FONT_BOLD, size=48,
              color="white", stroke_color="black", stroke_width=3.0)
        .set_duration(hook.duration)
    )

    hook_full = _fullscreen_zoom(hook)
    hook_comp = CompositeVideoClip(
        [hook_full, hook_label.set_position(("center", 800))],
        size=RESOLUTION,
    ).set_duration(hook.duration)

    # ===================================================================
    # 2. Éléments statiques du corps principal
    # ===================================================================
    bg = _create_background(full_dur)

    # Titre du clip (nettoyé)
    title_raw = clip_data.get("title", "Titre du clip")
    # Tronquer si trop long
    if len(title_raw) > 55:
        title_raw = title_raw[:52].strip() + "..."
    # Nettoyer les caractères problématiques
    title_clean = "".join(
        c for c in title_raw if c.isprintable()
    ).strip()

    title_clip = (
        _text(title_clean, font=FONT_BOLD, size=44,
              color="white", stroke_color="black", stroke_width=2.5)
        .set_duration(full_dur)
        .set_position(("center", 40))
    )

    # @streamer en bas
    streamer = clip_data.get("broadcaster_name", "Streamer")
    streamer_clip = (
        _text(f"@{streamer}", font=FONT_REGULAR, size=30,
              color="#AAAAAA", stroke_color="black", stroke_width=1.0)
        .set_duration(full_dur)
        .set_position(("center", 1860))
    )

    # Bandeau dégradé en haut
    gradient = _top_gradient(full_dur)

    # ===================================================================
    # 3. Gameplay fullscreen + webcam PIP
    # ===================================================================
    has_webcam = webcam_info.get("has_webcam", False)

    # Gameplay : toujours fullscreen zoom centré
    gameplay_zone = _fullscreen_zoom(clip_raw)

    # Ken Burns : zoom progressif léger
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

    # Composition de base
    base_elements = [bg, gameplay_clip, gradient, title_clip, streamer_clip]

    # PIP webcam
    if has_webcam:
        pip = _pip_webcam(clip_raw, webcam_info["bbox"], full_dur)
        if pip is not None:
            base_elements.append(pip)

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
    # 5. Sous-titres
    # ===================================================================
    subtitle_layers = []
    if subtitles:
        for group in subtitles:
            start = group.get("start", 0.0)
            end = group.get("end", 0.0)
            dur = end - start
            if dur < 0.1 or start >= full_dur:
                continue
            sc = _subtitle_clip(group, dur)
            if sc is not None:
                sc = sc.set_start(start).set_position(("center", 1320))
                subtitle_layers.append(sc)

    # ===================================================================
    # 6. Barre de progression + CTA
    # ===================================================================
    prog = _progress_bar(full_dur, RESOLUTION[0]).set_position(("center", 1916))

    cta_layers = []
    cta_points = [full_dur * 0.35, full_dur * 0.75]
    for ct in cta_points:
        cta = _cta_clip("ABONNE-TOI", ct, full_dur)
        cta_layers.append(cta)

    # ===================================================================
    # 7. Composition finale du corps
    # ===================================================================
    corp = CompositeVideoClip(
        base_elements + subtitle_layers + cta_layers + [prog],
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
        outro = VideoFileClip(END_VIDEO).resize(RESOLUTION)
        if outro.duration > 1.5:
            outro = outro.subclip(0, 1.5)
        end_clips.append(outro)

    final = concatenate_videoclips(end_clips)

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