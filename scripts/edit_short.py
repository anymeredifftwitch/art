# scripts/edit_short.py
"""
Montage video unifie pour Shorts YouTube (9:16).

Pipeline :
1.  Hook d'intro (teaser 2-3s les plus intenses)
2.  Split ecran (webcam + gameplay) ou plein ecran avec Ken Burns
3.  Sous-titres groupes
4.  Flashs sur pics audio
5.  Barre de progression
6.  Call-to-action animes (milieu + fin)
7.  Texte titre
8.  Sequence de fin
"""

import os
import numpy as np
from moviepy.editor import (
    VideoFileClip,
    CompositeVideoClip,
    TextClip,
    ImageClip,
    ColorClip,
    concatenate_videoclips,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
RESOLUTION = (1080, 1920)  # 9:16
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

FONT_BOLD = os.path.join(ASSETS_DIR, "Roboto-Bold.ttf")
FONT_REGULAR = os.path.join(ASSETS_DIR, "Roboto-Regular.ttf")
BG_IMAGE = os.path.join(ASSETS_DIR, "fond_short.png")
END_VIDEO = os.path.join(ASSETS_DIR, "fin_de_short.mp4")

for p in [FONT_BOLD, FONT_REGULAR]:
    if not os.path.exists(p):
        FONT_BOLD = "sans"
        FONT_REGULAR = "sans"
        break


# ---------------------------------------------------------------------------
# Petits clips reutilisables
# ---------------------------------------------------------------------------


def _text(text, font=FONT_BOLD, size=70, color="white",
          stroke_color="black", stroke_width=1.5):
    """TextClip securise (fallback si police absente)."""
    kwargs = dict(
        text=text,
        fontsize=size,
        color=color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
    )
    try:
        return TextClip(font=font, **kwargs)
    except Exception:
        return TextClip(**kwargs)


def _subtitle_clip(group, duration, target_w):
    """Un groupe de sous-titres : fond semi-transparent + texte blanc."""
    text = group.get("text", "")
    if not text.strip():
        return None

    # Fond semi-transparent
    bg = ColorClip((target_w, 90), color=(0, 0, 0)).set_opacity(0.50)

    # Texte centre
    tc = _text(text, font=FONT_BOLD, size=54, color="white", stroke_width=2.0)
    tc = tc.set_duration(duration)

    comp = CompositeVideoClip(
        [bg, tc.set_position("center")],
        size=(target_w, 90),
    ).set_duration(duration)
    return comp


def _progress_bar(duration, target_w):
    """Barre de progression fine (hauteur 6px) en bas."""

    def make_frame(t):
        frame = np.zeros((6, target_w, 3), dtype=np.uint8)
        progress = min(t / max(duration, 0.01), 1.0)
        filled = int(progress * target_w)
        frame[:, :filled] = [255, 255, 255]
        return frame

    from moviepy.editor import VideoClip
    return VideoClip(make_frame, duration=duration)


def _cta_clip(text, start_time, duration):
    """Call-to-action anime : scale-in, hold, scale-out."""

    def _pos(t):
        local_t = t - start_time
        if local_t < 0.15:
            s = local_t / 0.15
        elif local_t > 1.85:
            s = max(0.0, (2.0 - local_t) / 0.15)
        else:
            s = 1.0
        # Effet de petit rebond vers le haut en scale-in
        y_offset = (1.0 - s) * 120
        return (RESOLUTION[0] / 2, 960 + y_offset)

    tc = _text(text, font=FONT_BOLD, size=48, color="#FFD700", stroke_width=2.5)
    tc = tc.set_duration(duration).set_position(_pos)
    return tc


def _create_background(duration):
    """Fond 9:16 personnalise ou noir."""
    if os.path.exists(BG_IMAGE):
        return ImageClip(BG_IMAGE).resize(RESOLUTION).set_duration(duration)
    return ColorClip(RESOLUTION, color=(0, 0, 0)).set_duration(duration)


def _fullscreen_zoom(clip):
    """Zoom centre pour remplir 1080x1920 (garde l'audio)."""
    c = clip.resize(height=RESOLUTION[1])
    c = c.crop(
        x_center=c.w / 2,
        width=RESOLUTION[0],
        y_center=c.h / 2,
        height=RESOLUTION[1],
    )
    # Garder l'audio du clip d'origine
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
    print("Debut du montage unifie...")

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
        _text("MOMENT EPIQUE", font=FONT_BOLD, size=56,
              color="#FF4500", stroke_width=2.5)
        .set_duration(hook.duration)
    )

    hook_full = _fullscreen_zoom(hook)
    hook_comp = CompositeVideoClip(
        [hook_full, hook_label.set_position(("center", 600))],
        size=RESOLUTION,
    ).set_duration(hook.duration)

    # ===================================================================
    # 2. Corps principal - elements statiques
    # ===================================================================
    bg = _create_background(full_dur)

    title_text = clip_data.get("title", "Titre du clip")
    streamer = clip_data.get("broadcaster_name", "Streamer")

    title_clip = (
        _text(title_text, font=FONT_BOLD, size=64, stroke_width=2.0)
        .set_duration(full_dur)
        .set_position(("center", 60))
    )

    streamer_clip = (
        _text(f"@{streamer}", font=FONT_REGULAR, size=38, stroke_width=1.0)
        .set_duration(full_dur)
        .set_position(("center", 1820))
    )

    # ===================================================================
    # 3. Split ecran ou plein ecran
    # ===================================================================
    has_webcam = webcam_info.get("has_webcam", False)

    if has_webcam:
        bbox = webcam_info["bbox"]

        # Zone webcam en haut
        webcam_zone = clip_raw.crop(
            x1=bbox["x1"], y1=bbox["y1"],
            x2=bbox["x2"], y2=bbox["y2"]
        )
        webcam_h = int(RESOLUTION[1] * 0.30)
        webcam_zone = webcam_zone.resize(height=webcam_h)
        wx = (RESOLUTION[0] - webcam_zone.w) // 2
        webcam_pos = (wx, 190)

        # Zone gameplay en dessous
        game_h = int(RESOLUTION[1] * 0.60)
        gameplay_zone = clip_raw.crop(
            x1=0, y1=bbox["y2"],
            x2=clip_raw.w, y2=clip_raw.h
        )
        gameplay_zone = gameplay_zone.resize(height=game_h)
        gx = (RESOLUTION[0] - gameplay_zone.w) // 2
        gy = webcam_h + 200
        game_pos = (gx, gy)
    else:
        # Plein ecran zoom centre
        gameplay_zone = _fullscreen_zoom(clip_raw)
        game_pos = (0, 0)

    # Ken Burns : zoom progressif via decalage de position
    KB_ZOOM_AMOUNT = 0.04

    def _kb_pos(base_pos, t):
        zoom = 1.0 + KB_ZOOM_AMOUNT * (t / max(full_dur, 0.01))
        bx, by = base_pos
        offset_x = (zoom - 1.0) * RESOLUTION[0] / 2
        offset_y = (zoom - 1.0) * RESOLUTION[1] / 2
        return (bx - offset_x, by - offset_y)

    # On agrandit legerement le clip pour avoir de la marge
    gameplay_clip = gameplay_zone.resize(
        (int(gameplay_zone.w * (1.0 + KB_ZOOM_AMOUNT)),
         int(gameplay_zone.h * (1.0 + KB_ZOOM_AMOUNT)))
    )
    gameplay_clip = gameplay_clip.set_position(
        lambda t: _kb_pos(game_pos, t)
    ).set_duration(full_dur)

    # Composition de base
    base_elements = [bg, gameplay_clip, title_clip, streamer_clip]

    if has_webcam:
        webcam_clip = webcam_zone.set_position(webcam_pos).set_duration(full_dur)
        base_elements.insert(1, webcam_clip)

    # ===================================================================
    # 4. Flashs sur pics audio
    # ===================================================================
    for peak_t in peaks:
        if peak_t >= full_dur:
            continue
        flash = ColorClip(RESOLUTION, color=(255, 255, 255))
        flash = flash.set_duration(0.08).set_start(peak_t).set_opacity(0.20)
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
            sc = _subtitle_clip(group, dur, RESOLUTION[0])
            if sc is not None:
                sc = sc.set_start(start).set_position(("center", 1280))
                subtitle_layers.append(sc)

    # ===================================================================
    # 6. Barre de progression + CTA
    # ===================================================================
    prog = _progress_bar(full_dur, RESOLUTION[0]).set_position(("center", 1914))

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
    # 8. Sequence de fin + concatenation
    # ===================================================================
    end_clips = [hook_comp, corp]

    if os.path.exists(END_VIDEO):
        outro = VideoFileClip(END_VIDEO).resize(RESOLUTION)
        if outro.duration > 1.5:
            outro = outro.subclip(0, 1.5)
        end_clips.append(outro)

    final = concatenate_videoclips(end_clips)

    # ===================================================================
    # 9. Ecriture
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

    print(f"Montage termine : {output_path}")