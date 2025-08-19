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

# 👇 Déclaration explicite du chemin vers ImageMagick (modifiable si nécessaire)
change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})

# ------------------------------
# Configuration globale
# ------------------------------
RESOLUTION    = (1080, 1920)  # (width, height)
MAX_DURATION  = 180  # secondes
WEBCAM_COORDS = {'x1': 5, 'y1': 8, 'x2': 542, 'y2': 282}
ASSETS_DIR    = os.path.join(os.path.dirname(__file__), '..', 'assets')
OUTPUT_FILE   = None  # on écrira vers le chemin passé en argument

# ------------------------------
# Import OpenCV (avec message clair si absent)
# ------------------------------
try:
    import cv2
except Exception as e:
    cv2 = None
    _cv2_import_error = e

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
    # Découpage de la zone webcam et redimensionnement pour la placer en haut
    cam = clip.crop(
        x1=WEBCAM_COORDS['x1'], y1=WEBCAM_COORDS['y1'],
        x2=WEBCAM_COORDS['x2'], y2=WEBCAM_COORDS['y2']
    )
    cam = cam.resize(height=int(RESOLUTION[1] * 0.33))
    x_pos = (RESOLUTION[0] - cam.w) // 2
    return cam.set_position((x_pos, 0))

def extract_gameplay(clip):
    # Zone de jeu sous la webcam (d'après ton code existant)
    game = clip.crop(y1=WEBCAM_COORDS['y2'], y2=clip.h)
    game = game.resize(height=int(RESOLUTION[1] * 0.67))
    x_pos = (RESOLUTION[0] - game.w) // 2
    y_pos = int(RESOLUTION[1] * 0.33)
    return game.set_position((x_pos, y_pos))

def full_screen_clip(clip):
    """
    Zoom centré pour remplir totalement RESOLUTION à partir du centre du clip originel.
    On redimensionne par hauteur puis on crop center (simule un zoom vertical centré).
    """
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
    except Exception:
        # si la font n'est pas trouvée, MoviePy utilisera une font par défaut
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
# Détection du visage sur la 1ère frame (dans la zone webcam)
# ------------------------------
def is_face_in_webcam_zone(clip):
    """
    Retourne True si au moins un visage est détecté DANS la zone WEBCAM_COORDS
    sur la première frame du clip. Utilise OpenCV Haarcascade.
    """
    if cv2 is None:
        raise RuntimeError(
            "OpenCV (cv2) n'est pas installé. Ajoute 'opencv-python-headless' "
            "ou 'opencv-python' dans requirements.txt. "
            f"Import error: {_cv2_import_error}"
        )

    # Charger le cascade Haar frontal (fourni par opencv)
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    if not os.path.exists(cascade_path):
        raise RuntimeError(f"Haarcascade introuvable à {cascade_path}")

    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Récupérer la première frame (frame à t=0)
    try:
        frame0 = clip.get_frame(0.0)  # moviepy renvoie image RGB
    except Exception as e:
        # problème pour extraire la frame
        raise RuntimeError(f"Impossible d'extraire la première frame: {e}")

    # Extraire la ROI correspondant à la webcam
    x1, y1, x2, y2 = WEBCAM_COORDS['x1'], WEBCAM_COORDS['y1'], WEBCAM_COORDS['x2'], WEBCAM_COORDS['y2']

    # Clamp des coordonnées au cas où (sécurité)
    h, w = frame0.shape[0], frame0.shape[1]
    x1c = max(0, min(w - 1, x1))
    x2c = max(0, min(w, x2))
    y1c = max(0, min(h - 1, y1))
    y2c = max(0, min(h, y2))

    if x2c <= x1c or y2c <= y1c:
        # zone invalide -> considérer comme aucun visage
        return False

    roi_rgb = frame0[y1c:y2c, x1c:x2c]  # moviepy: array RGB
    # Convertir en niveaux de gris pour OpenCV (qui attend généralement BGR, mais conversion RGB->GRAY ok)
    roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)

    # Détection
    faces = face_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=5, minSize=(20, 20))

    return len(faces) > 0

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

    # Vérifier si visage présent dans la zone webcam (sur la 1ère frame)
    try:
        face_in_webcam = is_face_in_webcam_zone(clip)
    except RuntimeError as e:
        # Remonter l'erreur mais fermer le clip proprement
        clip.close()
        raise

    # Construire les éléments visuels et textes
    bg = create_background(duration)

    title_text = clip_data.get('title', 'Titre du clip')
    streamer   = clip_data.get('broadcaster_name', 'Streamer')
    title_clip = create_text_clip(title_text, "Roboto-Bold.ttf", 70, 1.5, 'top', duration)
    streamer_clip = create_text_clip(f"@{streamer}", "Roboto-Regular.ttf", 40, 0.5, 'bottom', duration)

    if face_in_webcam:
        # Comportement original : webcam visible + gameplay
        webcam = extract_webcam(clip)
        gameplay = extract_gameplay(clip)
        overlays = [gameplay, webcam, title_clip, streamer_clip]
        composed = CompositeVideoClip([bg] + overlays, size=RESOLUTION).set_audio(clip.audio)
    else:
        # Aucun visage dans la zone webcam : on ne montre PAS la webcam.
        # On zoom depuis le centre de la vidéo originelle pour remplir l'écran.
        gameplay_full = full_screen_clip(clip)
        overlays = [gameplay_full, title_clip, streamer_clip]
        # On met le gameplay zoomé en arrière-plan (pas besoin du bg séparé)
        composed = CompositeVideoClip(overlays, size=RESOLUTION).set_audio(clip.audio)

    final = append_end_sequence(composed)

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
    try:
        process_gameplay_clip(video_path, output, MAX_DURATION, {
            'title': title,
            'broadcaster_name': streamer,
            'game_name': game
        })
        print(f"✅ Fini : {output}")
    except Exception as e:
        print(f"❌ Erreur lors du traitement : {e}")
        raise
