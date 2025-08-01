import os
import sys
from typing import List, Optional

from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip, ImageClip, ColorClip, concatenate_videoclips
from moviepy.video.fx.all import crop, even_size, resize as moviepy_resize
import numpy as np

# ==============================================================================
# CONFIGURATION : DÉFINISSEZ VOS COORDONNÉES DE WEBCAM ICI
# Remplacez les valeurs ci-dessous par les coordonnées de votre propre webcam.
# Ces coordonnées sont pour une vidéo source en 1920x1080.
# ==============================================================================
WEBCAM_COORDS = {
    'x1': 5,
    'y1': 8,
    'x2': 542,
    'y2': 282
}

def trim_video_for_short(
    input_path: str,
    output_path: str,
    max_duration_seconds: int = 60,
    clip_data: Optional[dict] = None
):
    """
    Traite une vidéo pour le format Short (9:16) avec une gestion spécifique
    de la webcam et du contenu principal en fonction de la catégorie.

    - Coupe si elle dépasse la durée maximale.
    - Ajoute un fond personnalisé (ou noir si l'image n'est pas trouvée).
    - Si la catégorie n'est PAS 'Just Chatting':
        - Place la vidéo du haut (webcam rognée) à 33% de la hauteur.
        - Place la vidéo du bas (gameplay rogné/zoomé) à 66% de la hauteur.
    - Ajoute le titre du clip, le nom du streamer et une icône Twitch.
    - Ajoute une séquence de fin de 1.2s.

    Args:
        input_path (str): Chemin vers le fichier vidéo d'entrée.
        output_path (str): Chemin où sauvegarder le fichier vidéo de sortie.
        max_duration_seconds (int): Durée maximale du Short en secondes.
        clip_data (Optional[dict]): Dictionnaire contenant 'title', 'broadcaster_name', et 'game_name'.
    """
    print(f"✂️ Traitement vidéo : {input_path}")
    print(f"Durée maximale souhaitée : {max_duration_seconds} secondes.")
    if clip_data:
        print(f"Titre du clip : {clip_data.get('title', 'N/A')}")
        print(f"Streamer : {clip_data.get('broadcaster_name', 'N/A')}")
        print(f"Catégorie de jeu : {clip_data.get('game_name', 'N/A')}")

    if not os.path.exists(input_path):
        print(f"❌ Erreur : Le fichier d'entrée n'existe pas à {input_path}")
        return None

    clip = None
    end_clip = None
    composed_main_video_clip = None
    final_video = None

    try:
        clip = VideoFileClip(input_path)

        original_width, original_height = clip.size
        print(f"Résolution originale du clip : {original_width}x{original_height}")

        # --- Gérer la durée ---
        if clip.duration > max_duration_seconds:
            print(f"Le clip ({clip.duration:.2f}s) dépasse la durée maximale. Découpage à {max_duration_seconds}s.")
            clip = clip.subclip(0, max_duration_seconds)
        else:
            print(f"Le clip ({clip.duration:.2f}s) est déjà dans la limite de durée.")

        duration = clip.duration

        # --- Définir la résolution cible pour les Shorts (9:16) ---
        target_width, target_height = 1080, 1920

        # --- DÉFINITION DES CHEMINS DES ASSETS ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.abspath(os.path.join(script_dir, '..', 'assets'))
        twitch_icon_path = os.path.join(assets_dir, 'twitch_icon.png')
        custom_background_image_path = os.path.join(assets_dir, 'fond_short.png')
        end_short_video_path = os.path.join(assets_dir, 'fin_de_short.mp4')

        font_path_regular = os.path.join(assets_dir, 'Roboto-Regular.ttf')
        font_path_bold = os.path.join(assets_dir, 'Roboto-Bold.ttf')

        if not os.path.exists(font_path_regular):
            print(f"⚠️ Police '{font_path_regular}' non trouvée. Utilisation de la police par défaut de MoviePy pour le texte normal.")
            font_path_regular = "sans"
        if not os.path.exists(font_path_bold):
            print(f"⚠️ Police '{font_path_bold}' non trouvée. Utilisation de la police par défaut de MoviePy (bold) pour les titres.")
            font_path_bold = "sans"

        # --- FIN DE LA DÉFINITION DES CHEMINS ---

        all_video_elements = [] # Liste pour tous les éléments vidéo à composer

        # --- Configuration du fond personnalisé ---
        background_clip = None
        if not os.path.exists(custom_background_image_path):
            print(f"❌ Erreur : L'image de fond personnalisée '{os.path.basename(custom_background_image_path)}' est introuvable dans '{assets_dir}'.")
            print("Utilisation d'un fond noir par défaut.")
            background_clip = ColorClip(size=(target_width, target_height), color=(0,0,0)).set_duration(duration)
        else:
            print(f"✅ Création d'un fond personnalisé avec l'image : {os.path.basename(custom_background_image_path)}")
            try:
                background_clip = ImageClip(custom_background_image_path)
                background_clip = moviepy_resize(background_clip, newsize=(target_width, target_height))
                background_clip = background_clip.set_duration(duration)
            except Exception as e:
                print(f"❌ Erreur lors du chargement ou du traitement de l'image de fond : {e}")
                print("Utilisation d'un fond noir par défaut.")
                background_clip = ColorClip(size=(target_width, target_height), color=(0,0,0)).set_duration(duration)
        
        all_video_elements.append(background_clip.set_position(("center", "center")))

        # --- Traitement de la vidéo principale en fonction de la catégorie ---
        # --- LIGNE CORRIGÉE : Gérer le cas où 'game_name' est None. ---
        game_name = (clip_data.get("game_name") or "").lower()
        if "just chatting" in game_name or "juste de causer" in game_name:
            print("⏩ Mode 'Just Chatting' : Zoom sur la vidéo principale.")
            
            # LOGIQUE RETOURNÉE À LA VERSION ORIGINALE : Utilisation d'un facteur de zoom fixe de 2
            main_video_clip_display = clip.copy()
            main_video_display_width = int(target_width * 2) # Facteur de zoom 2
            main_video_clip_display = moviepy_resize(main_video_clip_display, width=main_video_display_width)
            main_video_clip_display = main_video_clip_display.fx(even_size)
            all_video_elements.append(main_video_clip_display.set_position(("center", "center")))
        else:
            print("✅ Mode 'Jeu' : Composition avec webcam et gameplay séparés.")
            
            # Définir la hauteur des deux sections
            webcam_section_height = int(target_height * 0.33)
            gameplay_section_height = target_height - webcam_section_height

            # 1. Préparer le clip de la webcam (haut)
            webcam_clip_original = crop(
                clip, 
                x1=WEBCAM_COORDS['x1'], 
                y1=WEBCAM_COORDS['y1'], 
                x2=WEBCAM_COORDS['x2'], 
                y2=WEBCAM_COORDS['y2']
            )
            webcam_clip_resized = moviepy_resize(webcam_clip_original, newsize=(target_width, webcam_section_height))
            all_video_elements.append(webcam_clip_resized.set_position(("center", "top")))

            # 2. Préparer le clip du gameplay (bas)
            # Logique de zoom corrigée pour la partie gameplay
            gameplay_clip_original_zoomed = moviepy_resize(clip, newsize=(None, gameplay_section_height))
            gameplay_clip_final = crop(
                gameplay_clip_original_zoomed,
                width=target_width,
                height=gameplay_section_height,
                x_center='center'
            )
            all_video_elements.append(gameplay_clip_final.set_position(("center", webcam_section_height)))


        # --- Fin du traitement de la vidéo principale ---

        video_with_visuals = CompositeVideoClip(all_video_elements, size=(target_width, target_height)).set_duration(duration)

        title_text = clip_data.get('title', 'Titre du clip')
        streamer_name = clip_data.get('broadcaster_name', 'Nom du streamer')

        # Ajout des textes et de l'icône Twitch
        text_color = "white"
        stroke_color = "black"
        stroke_width = 1.5
        
        title_clip = TextClip(title_text, fontsize=70, color=text_color,
                              font=font_path_bold, stroke_color=stroke_color, stroke_width=stroke_width,
                              size=(target_width * 0.9, None),
                              method='caption') \
                     .set_duration(duration) \
                     .set_position(("center", int(target_height * 0.08)))

        stroke_width = 0.5
        streamer_clip = TextClip(f"@{streamer_name}", fontsize=40, color=text_color,
                                 font=font_path_regular, stroke_color=stroke_color, stroke_width=stroke_width) \
                        .set_duration(duration) \
                        .set_position(("center", int(target_height * 0.85) - 40)) 
        
        twitch_icon_clip = None
        if os.path.exists(twitch_icon_path):
            try:
                twitch_icon_clip = ImageClip(twitch_icon_path, duration=duration)
                twitch_icon_clip = moviepy_resize(twitch_icon_clip, width=80)
                
                icon_x = title_clip.pos[0] - twitch_icon_clip.w - 10
                icon_y = title_clip.pos[1] + (title_clip.h / 2) - (twitch_icon_clip.h / 2)

                twitch_icon_clip = twitch_icon_clip.set_position((icon_x, icon_y))
                print("✅ Icône Twitch ajoutée.")
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement ou du traitement de l'icône Twitch : {e}. L'icône ne sera pas ajoutée.")
                twitch_icon_clip = None
        else:
            print("⚠️ Fichier 'twitch_icon.png' non trouvé dans le dossier 'assets'. L'icône ne sera pas ajoutée.")

        final_elements_main_video = [video_with_visuals, title_clip, streamer_clip]
        if twitch_icon_clip:
            final_elements_main_video.append(twitch_icon_clip)

        composed_main_video_clip = CompositeVideoClip(final_elements_main_video).set_duration(duration)

        # --- AJOUT DE LA SÉQUENCE DE FIN ---
        print(f"⏳ Ajout de la séquence de fin : {os.path.basename(end_short_video_path)}")
        if os.path.exists(end_short_video_path):
            try:
                end_clip = VideoFileClip(end_short_video_path)
                end_clip = moviepy_resize(end_clip, newsize=(target_width, target_height))
                
                if end_clip.duration > 1.2:
                    end_clip = end_clip.subclip(0, 1.2)
                elif end_clip.duration < 1.2:
                    print(f"⚠️ La vidéo de fin ({end_clip.duration:.2f}s) est plus courte que 1.2s. Elle ne sera pas étirée.")
                
                final_video = concatenate_videoclips([composed_main_video_clip, end_clip])
                print("✅ Séquence de fin ajoutée avec succès.")

            except Exception as e:
                print(f"❌ Erreur lors du chargement ou du traitement de la vidéo de fin : {e}. Le Short sera créé sans séquence de fin.")
                final_video = composed_main_video_clip
        else:
            print(f"⚠️ Fichier 'fin_de_short.mp4' non trouvé dans le dossier 'assets'. Le Short sera créé sans séquence de fin.")
            final_video = composed_main_video_clip

        final_video.write_videofile(output_path,
                                    codec="libx264",
                                    audio_codec="aac",
                                    temp_audiofile='temp-audio.m4a',
                                    remove_temp=True,
                                    fps=clip.fps,
                                    logger=None)
        print(f"✅ Clip traité et sauvegardé : {output_path}")
        return output_path
            
    except Exception as e:
        print(f"❌ Erreur CRITIQUE lors du traitement vidéo : {e}")
        print("Assurez-vous que 'ffmpeg' est installé et accessible dans votre PATH, et que tous les assets sont valides.")
        print("Pour l'installer: https://ffmpeg.org/download.html")
        return None
    finally:
        if 'clip' in locals() and clip is not None:
            clip.close()
        if 'composed_main_video_clip' in locals() and composed_main_video_clip is not None:
            composed_main_video_clip.close()
        if 'end_clip' in locals() and end_clip is not None:
            end_clip.close()
        if 'final_video' in locals() and final_video is not None:
            final_video.close()

def trim_video_for_short_test(
    input_path: str,
    output_path: str,
    is_webcam_clip: bool,
    max_duration_seconds: int = 60,
):
    """
    Fonction de test pour vérifier la composition vidéo avec ou sans webcam.
    Args:
        input_path (str): Chemin vers le fichier vidéo d'entrée.
        output_path (str): Chemin où sauvegarder le fichier vidéo de sortie.
        is_webcam_clip (bool): True pour simuler un clip avec webcam (jeu), False pour simuler un clip sans (just chatting).
        max_duration_seconds (int): Durée maximale du Short en secondes.
    """
    
    clip_info_test = {
        'title': 'Test de Composition Vidéo',
        'broadcaster_name': 'AnymeRediffTwitch',
        'game_name': 'Gaming' if is_webcam_clip else 'Just Chatting'
    }
    
    print(f"\n--- Démarrage du traitement de TEST pour {input_path} ---")
    print(f"Mode de test : {'Avec webcam (jeu)' if is_webcam_clip else 'Sans webcam (just chatting)'}")
    
    # Note: L'appel à trim_video_for_short_test dans le bloc if __name__ == "__main__":
    # doit utiliser la variable 'output_video_file_game' qui est définie dans le même bloc.
    # Je vais donc corriger cette partie pour que les deux tests fonctionnent correctement.
    # La correction est appliquée dans l'appel ci-dessous.
    processed_file = trim_video_for_short(
        input_path=input_path,
        output_path=output_path, # Utilisation de l'output_path passé en argument
        max_duration_seconds=max_duration_seconds,
        clip_data=clip_info_test
    )
    
    if processed_file:
        print(f"\n🎉 Test réussi ! Vidéo traitée et sauvegardée : {processed_file}")
    else:
        print("\n❌ Le test de traitement vidéo a échoué.")

# --- Exemple d'utilisation du script en local ---
if __name__ == "__main__":
    output_dir = "output_test"
    os.makedirs(output_dir, exist_ok=True)
    
    input_video_file = "video.mp4" # Assurez-vous que ce fichier existe dans le même dossier que ce script

    # --- Test 1 : Clip de jeu (avec webcam) ---
    output_video_file_game = os.path.join(output_dir, "short_game_test.mp4")
    trim_video_for_short_test(
        input_path=input_video_file,
        output_path=output_video_file_game,
        is_webcam_clip=True
    )
    
    # --- Test 2 : Clip Just Chatting (sans webcam) ---
    output_video_file_chat = os.path.join(output_dir, "short_chat_test.mp4")
    trim_video_for_short_test(
        input_path=input_video_file,
        output_path=output_video_file_chat,
        is_webcam_clip=False
    )