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
    CompositeAudioClip,
    AudioFileClip,
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
    """
    Sous-titre style TikTok : texte blanc sur fond semi-transparent.
    Pas de stroke épais qui rend le texte fantôme.
    """
    text = group.get("text", "")
    if not text.strip():
        return None

    # Texte blanc pur, sans contour (le fond pill suffit pour la lisibilité)
    tc = _text(
        text,
        font=FONT_BOLD,
        size=56,
        color="white",
        stroke_color="white",
        stroke_width=0,
    )
    tc = tc.set_duration(duration)

    # Fond pill : bande noire semi-transparente derrière le texte
    bar_h = int(tc.h * 1.3) if tc.h else 76
    bar = ColorClip((RESOLUTION[0], bar_h), color=(0, 0, 0))
    bar = bar.set_duration(duration).set_opacity(0.45)

    comp = CompositeVideoClip(
        [bar.set_position(("center", 0)),
         tc.set_position(("center", "center"))],
        size=(RESOLUTION[0], bar_h),
    ).set_duration(duration)
    return comp


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


def _title_banner(duration, text):
    """
    Bandeau titre style TikTok : fond noir quasi-opaque + texte blanc large.
    Renvoie un CompositeVideoClip centré en haut.
    """
    h = 140
    # Fond noir quasi-opaque
    bar = ColorClip((RESOLUTION[0], h), color=(0, 0, 0))
    bar = bar.set_duration(duration).set_opacity(0.85)

    # Accent : fine ligne rouge en bas du bandeau
    accent = ColorClip((RESOLUTION[0], 4), color=(229, 9, 20))
    accent = accent.set_duration(duration)

    # Texte blanc, plus gros, stroke fin
    tc = _text(text, font=FONT_BOLD, size=52,
               color="white", stroke_color="black", stroke_width=0.8)
    tc = tc.set_duration(duration).set_position(("center", "center"))

    comp = CompositeVideoClip(
        [bar.set_position((0, 0)),
         accent.set_position((0, h - 4)),
         tc.set_position(("center", "center"))],
        size=(RESOLUTION[0], h),
    ).set_duration(duration)
    return comp.set_position((0, 0))


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

    # Hook title: reuse the AI-generated overlay_title as hook text
    hook_title = clip_data.get("overlay_title", "")
    hook_title = "".join(c for c in hook_title if c.isprintable()).strip()
    if not hook_title or len(hook_title) < 3:
        hook_title = "🔥 BEST MOMENT"

    # TTS voiceover that reads the title during the hook (TikTok style)
    hook_tts_audio = _generate_tts(hook_title, hook.duration)

    # Fond pill opaque derrière le hook
    hook_pill_h = 90
    # Wider pill for longer titles
    hook_pill_w = min(900, max(500, len(hook_title) * 18))
    hook_pill = (
        ColorClip((hook_pill_w, hook_pill_h), color=(0, 0, 0))
        .set_duration(hook.duration)
        .set_opacity(0.80)
    )
    # Accent rouge à gauche du pill
    hook_accent = (
        ColorClip((6, hook_pill_h), color=(229, 9, 20))
        .set_duration(hook.duration)
    )
    hook_label = (
        _text(hook_title, font=FONT_BOLD, size=42,
              color="white", stroke_color="black", stroke_width=0.8)
        .set_duration(hook.duration)
        .set_position(("center", "center"))
    )
    hook_badge = CompositeVideoClip(
        [hook_pill.set_position((0, 0)),
         hook_accent.set_position((0, 0)),
         hook_label.set_position(("center", "center"))],
        size=(hook_pill_w, hook_pill_h),
    ).set_duration(hook.duration)

    hook_full = _fullscreen_zoom(hook)
    hook_comp = CompositeVideoClip(
        [hook_full, hook_badge.set_position(("center", 760))],
        size=RESOLUTION,
    ).set_duration(hook.duration)

    # Add TTS voiceover to the hook
    if hook_tts_audio is not None:
        try:
            hook_video_audio = hook.audio if hook.audio is not None else hook_full.audio
            if hook_video_audio is not None:
                # Mix: original audio lowered + TTS on top
                mixed = CompositeAudioClip([
                    hook_video_audio.volumex(0.3),
                    hook_tts_audio,
                ])
                hook_comp = hook_comp.set_audio(mixed)
            else:
                hook_comp = hook_comp.set_audio(hook_tts_audio)
        except Exception as e:
            print(f"⚠️  Impossible de mixer le TTS : {e}")

    # ===================================================================
    # 2. Éléments statiques du corps principal
    # ===================================================================
    bg = _create_background(full_dur)

    # Titre overlay : généré par l'IA à partir de la transcription
    overlay_title = clip_data.get("overlay_title", "")
    # Nettoyer
    overlay_title = "".join(c for c in overlay_title if c.isprintable()).strip()
    if len(overlay_title) > 42:
        overlay_title = overlay_title[:39].strip() + "..."

    title_banner = None
    if overlay_title:
        title_banner = _title_banner(full_dur, overlay_title)

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
    # NOTE: on n'ajoute PAS de nom de streamer ni de CTA :
    # le clip Twitch brut contient déjà ces overlays.
    # On garde uniquement le titre overlay (si généré) + barre de progression.
    base_elements = [bg, gameplay_clip]
    if title_banner is not None:
        base_elements.append(title_banner)

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
    # 5. Sous-titres (position basse pour éviter les overlays Twitch)
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
                sc = sc.set_start(start).set_position(("center", 1350))
                subtitle_layers.append(sc)

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