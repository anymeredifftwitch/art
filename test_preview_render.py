# test_preview_render.py
"""
Générateur de rendu visuel comparatif AVANT vs APRÈS (0s et 5s)
Permet de visualiser le nouveau design sans avoir à exporter de vidéo complète.
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

FONT_BOLD_PATH = os.path.join(ASSETS_DIR, "OpenSans-Bold.ttf")
FONT_REGULAR_PATH = os.path.join(ASSETS_DIR, "OpenSans-Regular.ttf")

RESOLUTION = (1080, 1920)


def create_title_badge_image(
    text,
    max_width=920,
    font_path=FONT_BOLD_PATH,
    font_size=42,
    bg_color=(255, 255, 255, 255),
    text_color=(15, 15, 15, 255),
    radius=26,
    padding_x=40,
    padding_y=24,
    line_spacing=10,
):
    """
    Crée une image RGBA avec le titre sous forme de bulle blanche arrondie
    avec word-wrap automatique (retour à la ligne) et centrage parfait.
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    words = text.strip().split()
    if not words:
        words = ["TITRE"]

    usable_width = max_width - (2 * padding_x)

    # Word wrap intelligent
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
        # Hauteur basée sur l'ascent/descent pour un espacement régulier
        line_heights.append(font_size)

    max_line_w = max(line_widths) if line_widths else 100
    total_text_h = sum(line_heights) + (len(lines) - 1) * line_spacing

    badge_w = min(max_width, max(400, max_line_w + 2 * padding_x))
    badge_h = total_text_h + 2 * padding_y

    img = Image.new("RGBA", (badge_w, badge_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rectangle blanc arrondi (fond badge TikTok)
    draw.rounded_rectangle(
        [(0, 0), (badge_w, badge_h)],
        radius=radius,
        fill=bg_color,
    )

    # Rendu de chaque ligne centrée
    current_y = padding_y
    for i, line in enumerate(lines):
        line_w = line_widths[i]
        line_x = (badge_w - line_w) // 2
        bbox = font.getbbox(line)
        # Décalage d'alignement fin
        offset_y = bbox[1]
        draw.text((line_x, current_y - offset_y), line, font=font, fill=text_color)
        current_y += line_heights[i] + line_spacing

    return img


def draw_subtitle(draw, text, y_pos, font_path=FONT_BOLD_PATH, font_size=46):
    """Dessine le sous-titre style TikTok (fond sombre semi-transparent + texte blanc)."""
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    bar_h = int(text_h * 1.6)
    bar_y1 = y_pos - bar_h // 2
    bar_y2 = y_pos + bar_h // 2

    # Bandeau sombre plein large
    draw.rectangle([(0, bar_y1), (RESOLUTION[0], bar_y2)], fill=(0, 0, 0, 140))
    # Texte blanc centré
    text_x = (RESOLUTION[0] - text_w) // 2
    draw.text((text_x, y_pos - text_h // 2 - bbox[1]), text, font=font, fill=(255, 255, 255, 255))


def create_simulated_streamer_bg():
    """Crée un fond de stream stylisé réaliste (caméra + setup ambiance néon/gaming)."""
    img = Image.new("RGBA", RESOLUTION, (25, 22, 35, 255))
    draw = ImageDraw.Draw(img)

    # Dégradé et ambiance de chambre gaming
    for y in range(RESOLUTION[1]):
        r = int(35 + (y / RESOLUTION[1]) * 20)
        g = int(30 + (y / RESOLUTION[1]) * 15)
        b = int(45 + (y / RESOLUTION[1]) * 25)
        draw.line([(0, y), (RESOLUTION[0], y)], fill=(r, g, b, 255))

    # Étagère et néons d'ambiance
    draw.line([(50, 250), (1030, 250)], fill=(255, 80, 180, 200), width=6)
    draw.line([(50, 400), (1030, 400)], fill=(80, 180, 255, 200), width=4)

    # Silhouette streamer
    draw.ellipse([(240, 500), (840, 1100)], fill=(60, 50, 70, 255))  # Tête/buste
    draw.ellipse([(340, 560), (740, 960)], fill=(220, 180, 160, 255))  # Visage
    draw.rectangle([(200, 1000), (880, 1800)], fill=(30, 30, 35, 255))  # Corps / T-shirt

    return img


def render_all_mockups():
    sample_title = "POV: Titre putaclic - Titre Putaclic - Il ne doit pas être coupé !"
    sample_sub = "gros dédicace, il"

    bg_base = create_simulated_streamer_bg()

    # =========================================================================
    # 1. AVANT (0s) : Titre laid bandeau noir coupé au milieu (Y=760)
    # =========================================================================
    img_avant_0s = bg_base.copy()
    draw_a0 = ImageDraw.Draw(img_avant_0s)
    # Fond noir au milieu
    draw_a0.rectangle([(150, 740), (930, 830)], fill=(0, 0, 0, 220))
    # Ligne rouge d'accent
    draw_a0.rectangle([(150, 740), (156, 830)], fill=(229, 9, 20, 255))
    # Texte rogné / coupé
    try:
        f_serif = ImageFont.truetype("times.ttf", 36)
    except Exception:
        f_serif = ImageFont.load_default()
    draw_a0.text((170, 765), "OV: IL M'A TROMPÉ EN LIV...", font=f_serif, fill=(255, 255, 255, 255))
    # Élément bas
    draw_a0.rounded_rectangle([(300, 1750), (780, 1830)], radius=20, fill=(185, 75, 45, 255))

    # =========================================================================
    # 2. AVANT (5s) : Titre bandeau noir haut (Y=0) + Sous-titres trop bas (Y=1450)
    # =========================================================================
    img_avant_5s = bg_base.copy()
    draw_a5 = ImageDraw.Draw(img_avant_5s)
    # Bandeau noir plein haut
    draw_a5.rectangle([(0, 0), (RESOLUTION[0], 140)], fill=(0, 0, 0, 225))
    draw_a5.rectangle([(0, 136), (RESOLUTION[0], 140)], fill=(229, 9, 20, 255))
    draw_a5.text((320, 50), "POV: IL M'A TROMPÉ EN LIVE", font=f_serif, fill=(255, 255, 255, 255))
    # Sous-titres trop bas (Y=1450 / 76%)
    draw_subtitle(draw_a5, sample_sub, y_pos=1450)
    # Élément bas
    draw_a5.rounded_rectangle([(300, 1750), (780, 1830)], radius=20, fill=(185, 75, 45, 255))

    # =========================================================================
    # 3. APRÈS (0s) : Bulle blanche TikTok en haut (Y=100) multiline
    # =========================================================================
    img_apres_0s = bg_base.copy()
    badge_0s = create_title_badge_image(sample_title)
    # Centrer le badge en haut (Y = 100)
    badge_x = (RESOLUTION[0] - badge_0s.width) // 2
    badge_y = 100
    img_apres_0s.paste(badge_0s, (badge_x, badge_y), badge_0s)
    # Élément bas
    draw_ap0 = ImageDraw.Draw(img_apres_0s)
    draw_ap0.rounded_rectangle([(300, 1750), (780, 1830)], radius=20, fill=(185, 75, 45, 255))

    # =========================================================================
    # 4. APRÈS (5s) : MÊME bulle blanche en haut (Y=100) + Sous-titres remontés (Y=1150 / 60%)
    # =========================================================================
    img_apres_5s = bg_base.copy()
    # MÊME badge en haut (position fixe t=0s -> fin)
    img_apres_5s.paste(badge_0s, (badge_x, badge_y), badge_0s)
    # Sous-titres remontés à 60% (Y=1150)
    draw_ap5 = ImageDraw.Draw(img_apres_5s)
    draw_subtitle(draw_ap5, sample_sub, y_pos=1150)
    # Élément bas (zone basse parfaitement dégagée)
    draw_ap5.rounded_rectangle([(300, 1750), (780, 1830)], radius=20, fill=(185, 75, 45, 255))

    # Sauvegarder les images individuelles
    p_a0 = os.path.join(DATA_DIR, "preview_avant_0s.png")
    p_a5 = os.path.join(DATA_DIR, "preview_avant_5s.png")
    p_ap0 = os.path.join(DATA_DIR, "preview_apres_0s.png")
    p_ap5 = os.path.join(DATA_DIR, "preview_apres_5s.png")

    img_avant_0s.save(p_a0)
    img_avant_5s.save(p_a5)
    img_apres_0s.save(p_ap0)
    img_apres_5s.save(p_ap5)

    # =========================================================================
    # 5. Rendu Karaoké Mot par Mot
    # =========================================================================
    img_karaoke = bg_base.copy()
    img_karaoke.paste(badge_0s, (badge_x, badge_y), badge_0s)
    # Rendu sous-titre karaoké avec "GROS" en blanc et "DÉDICACE" en jaune actif
    sys.path.append(os.path.join(BASE_DIR, "scripts"))
    import edit_short
    sub_img = edit_short._render_karaoke_subtitle_image(
        [{"word": "GROS"}, {"word": "DÉDICACE,"}, {"word": "IL"}, {"word": "A"}],
        active_idx=1,
    )
    if sub_img:
        sub_x = (RESOLUTION[0] - sub_img.width) // 2
        img_karaoke.paste(sub_img, (sub_x, 1150), sub_img)

    p_karaoke = os.path.join(DATA_DIR, "preview_karaoke_subtitles.png")
    img_karaoke.save(p_karaoke)

    # =========================================================================
    # 6. Rendu Split-Screen Gameplay + Webcam
    # =========================================================================
    img_split = Image.new("RGBA", RESOLUTION, (15, 15, 20, 255))
    draw_split = ImageDraw.Draw(img_split)
    # Zone Webcam haut (0 à 720)
    draw_split.rectangle([(0, 0), (1080, 720)], fill=(32, 38, 52, 255))
    # Simuler streamer webcam
    draw_split.ellipse([(440, 260), (640, 480)], fill=(210, 160, 130, 255)) # Visage
    draw_split.ellipse([(340, 480), (740, 720)], fill=(45, 55, 75, 255))   # Buste
    # Ligne de séparation
    draw_split.rectangle([(0, 718), (1080, 722)], fill=(15, 15, 15, 255))
    # Zone Gameplay bas (720 à 1920)
    draw_split.rectangle([(0, 722), (1080, 1920)], fill=(22, 26, 35, 255))
    # Simuler décor gameplay FPS / GTA
    draw_split.polygon([(100, 1400), (540, 950), (980, 1400)], fill=(40, 48, 65, 255))
    draw_split.rectangle([(480, 1100), (600, 1220)], outline=(255, 230, 0, 200), width=3) # Réticule/Focus

    # Incruster Badge titre et Karaoké
    img_split.paste(badge_0s, (badge_x, 90), badge_0s)
    if sub_img:
        img_split.paste(sub_img, (sub_x, 1150), sub_img)

    # Barre de progression
    draw_split.rectangle([(0, 1916), (420, 1920)], fill=(220, 220, 220, 255))

    p_split = os.path.join(DATA_DIR, "preview_splitscreen_gameplay.png")
    img_split.save(p_split)

    # =========================================================================
    # 7. Créer la planche comparative 2x2 haute définition
    # =========================================================================
    thumb_w, thumb_h = 540, 960
    header_h = 100
    board_w = thumb_w * 2 + 60
    board_h = (thumb_h + header_h) * 2 + 60

    board = Image.new("RGBA", (board_w, board_h), (245, 245, 248, 255))
    draw_b = ImageDraw.Draw(board)

    try:
        f_title = ImageFont.truetype(FONT_BOLD_PATH, 36)
        f_sub = ImageFont.truetype(FONT_REGULAR_PATH, 22)
    except Exception:
        f_title = ImageFont.load_default()
        f_sub = ImageFont.load_default()

    panels = [
        ("1. AVANT : Bandeau noir & Titre coupé", img_avant_0s, 20, 20),
        ("2. APRÈS : Bulle TikTok fixe (Open Sans)", img_apres_5s, thumb_w + 40, 20),
        ("3. NOUVEAU : Sous-titres Karaoké Mot par Mot", img_karaoke, 20, thumb_h + header_h + 40),
        ("4. NOUVEAU : Mode Split-Screen Gameplay", img_split, thumb_w + 40, thumb_h + header_h + 40),
    ]

    for label, img_src, px, py in panels:
        draw_b.text((px + 10, py + 25), label, font=f_title, fill=(20, 20, 25, 255))
        thumb = img_src.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        board.paste(thumb, (px, py + header_h))
        draw_b.rectangle([(px, py + header_h), (px + thumb_w, py + header_h + thumb_h)], outline=(200, 200, 210, 255), width=2)

    board_path = os.path.join(DATA_DIR, "comparatif_avant_apres.png")
    board.save(board_path)

    print(f"✅ Images générées avec succès :")
    print(f"  - {p_a0}")
    print(f"  - {p_a5}")
    print(f"  - {p_ap0}")
    print(f"  - {p_ap5}")
    print(f"  - {p_karaoke}")
    print(f"  - {p_split}")
    print(f"  - Planche comparative : {board_path}")
    return board_path


if __name__ == "__main__":
    render_all_mockups()
