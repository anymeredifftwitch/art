# scripts/process_video_gameplay.py

import sys
import os

from moviepy.editor import (
    VideoFileClip,
    CompositeVideoClip,
    TextClip,
    ImageClip,
    concatenate_videoclips,
    ColorClip
)
from moviepy.video.fx.resize import resize
from moviepy.config import change_settings

# 👇 Déclaration explicite du chemin vers ImageMagick
change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})

# ------------------------------
# Configuration globale
# ------------------------------
RESOLUTION    = (1080, 1920)
MAX_DURATION  = 180  # secondes
WEBCAM_COORDS = {'x1': 5, 'y1': 8, 'x2': 542, 'y2': 282}
ASSETS_DIR    = os.path.join(os.path.dirname(__file__), '..', 'assets')
OUTPUT_FILE   = None  # on écrira vers le chemin passé en argument

# ------------------------------
# Fonctions utilitaires
# ------------------------------
def load_clip(path):
    clip = VideoFileClip(path)
    if clip.duration > MAX_DURATION:
        clip = clip.subclip(0, MAX_DURATION)
    return clip

def create_background(duration):
    bg_path = os.path.join(ASSETS_DIR, "fond_short.png")
    if os.path.exists(bg_path):
        bg = ImageClip(bg_path).resize(RESOLUTION)
    else:
        bg = ColorClip(RESOLUTION, color=(0, 0, 0))
    return bg.set_duration(duration)

def extract_webcam(clip):
    cam = clip.crop(
        x1=WEBCAM_COORDS['x1'], y1=WEBCAM_COORDS['y1'],
        x2=WEBCAM_COORDS['x2'], y2=WEBCAM_COORDS['y2']
    )
    cam = cam.resize(height=int(RESOLUTION[1] * 0.33))
    x_pos = (RESOLUTION[0] - cam.w) // 2
    return cam.set_position((x_pos, 0))

def extract_gameplay(clip):
    game = clip.crop(y1=WEBCAM_COORDS['y2'], y2=clip.h)
    game = game.resize(height=int(RESOLUTION[1] * 0.67))
    x_pos = (RESOLUTION[0] - game.w) // 2
    y_pos = int(RESOLUTION[1] * 0.33)
    return game.set_position((x_pos, y_pos))

def full_screen_clip(clip):
    c = clip.resize(height=RESOLUTION[1])
    c = c.crop(
        x_center=c.w/2, width=RESOLUTION[0],
        y_center=c.h/2, height=RESOLUTION[1]
    )
    return c.set_position((0, 0))

def create_text_clip(text, font, size, stroke, y_pos, duration):
    try:
        txt = TextClip(
            text, fontsize=size, font=font,
            color='white', stroke_color='black', stroke_width=stroke
        )
    except:
        txt = TextClip(
            text, fontsize=size,
            color='white', stroke_color='black', stroke_width=stroke
        )
    return txt.set_position(('center', y_pos)).set_duration(duration)

def append_end_sequence(main_clip):
    end_path = os.path.join(ASSETS_DIR, "fin_de_short.mp4")
    if os.path.exists(end_path):
        end_clip = VideoFileClip(end_path).resize(RESOLUTION)
        return concatenate_videoclips([main_clip, end_clip])
    return main_clip

# ------------------------------
# Fonction principale exposée
# ------------------------------
def process_gameplay_clip(input_path, output_path, max_duration_seconds, clip_data):
    """
    input_path: chemin vers le MP4 brut
    output_path: chemin où enregistrer le Short final
    clip_data doit contenir 'title', 'broadcaster_name', 'game_name'
    """
    # On ignore max_duration_seconds ici, on utilise MAX_DURATION
    clip = load_clip(input_path)
    duration = clip.duration
    bg = create_background(duration)

    # Layout gameplay : webcam + zone de jeu
    webcam = extract_webcam(clip)
    gameplay = extract_gameplay(clip)
    overlays = [gameplay, webcam]

    # Textes
    title_text = clip_data.get('title', 'Titre du clip')
    streamer   = clip_data.get('broadcaster_name', 'Streamer')
    title_clip = create_text_clip(title_text, "Roboto-Bold.ttf", 70, 1.5, 'top', duration)
    streamer_clip = create_text_clip(f"@{streamer}", "Roboto-Regular.ttf", 40, 0.5, 'bottom', duration)
    overlays.extend([title_clip, streamer_clip])

    # Composition
    composed = CompositeVideoClip([bg] + overlays, size=RESOLUTION).set_audio(clip.audio)
    final    = append_end_sequence(composed)

    # Écriture du fichier
    final.write_videofile(
        output_path,
        fps=30,
        codec="libx264",
        audio_codec="aac"
    )

    # Fermer les clips pour libérer la mémoire
    clip.close()
    composed.close()
    final.close()

    return output_path

# ------------------------------
# Entrée en mode standalone (facultatif)
# ------------------------------
if __name__ == "__main__":
    if len(sys.argv) == 5:
        video_path, title, streamer, game = sys.argv[1:]
    else:
        # Test local
        video_path = "video.mp4"
        title      = "Test de montage de clip"
        streamer   = "Anyme023"
        game       = "Valorant"

    output = "output.mp4"
    process_gameplay_clip(video_path, output, MAX_DURATION, {
        'title': title,
        'broadcaster_name': streamer,
        'game_name': game
    })
    print(f"✅ Fini : {output}")
