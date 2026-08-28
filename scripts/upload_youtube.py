# scripts/upload_youtube.py
import os
import sys
import json
import google.oauth2.credentials
import google_auth_oauthlib.flow
import google.auth.transport.requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

# L'API scope nécessaire pour uploader des vidéos
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'

# Chemin vers le fichier client_secret.json
CLIENT_SECRETS_FILE = 'client_secret.json'
# Le fichier token.json
TOKEN_FILE = 'token.json'

def get_authenticated_service():
    """
    Authentifie l'utilisateur et retourne un objet de service YouTube.
    Gère le flux OAuth 2.0 et stocke les jetons d'accès.
    """
    credentials = None
    # Charger les jetons d'accès existants s'ils sont disponibles
    if os.path.exists(TOKEN_FILE):
        try:
            credentials = google.oauth2.credentials.Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception as e:
            print(f"⚠️  Erreur lecture token.json : {e}")

    # Si les jetons ne sont pas valides ou n'existent pas, lancer le flux d'authentification
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            print("🔑 Rafraîchissement du jeton d'accès YouTube...")
            try:
                credentials.refresh(google.auth.transport.requests.Request())
            except Exception as e:
                print(f"⚠️  Erreur rafraîchissement token YouTube : {e}")
                credentials = None

        if not credentials:
            if not os.path.exists(CLIENT_SECRETS_FILE):
                raise FileNotFoundError(f"Le fichier {CLIENT_SECRETS_FILE} ou {TOKEN_FILE} est requis pour l'upload YouTube.")

            # Vérifier si l'environnement est interactif
            if not sys.stdin.isatty():
                raise RuntimeError("Environnement non-interactif (CI/Actions) : token.json manquant ou invalide.")

            print("🔑 Lancement du flux d'authentification YouTube...")
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
            auth_url, _ = flow.authorization_url(prompt='consent')
            print(f"Veuillez ouvrir ce lien dans votre navigateur et autoriser l'application:\n{auth_url}")
            code = input("Entrez le code de vérification ici: ").strip()
            flow.fetch_token(code=code)
            credentials = flow.credentials

        # Sauvegarder les jetons pour les exécutions futures
        try:
            with open(TOKEN_FILE, 'w') as token:
                token.write(credentials.to_json())
            print("✅ Jeton d'accès YouTube sauvegardé.")
        except Exception:
            pass

    return build(API_SERVICE_NAME, API_VERSION, credentials=credentials)

def upload_youtube_short(youtube_service, video_path, metadata):
    """
    Uploade un fichier vidéo sur YouTube en tant que Short.

    Args:
        youtube_service: L'objet de service YouTube authentifié.
        video_path (str): Chemin vers le fichier vidéo à uploader.
        metadata (dict): Dictionnaire contenant le titre, la description, les tags, etc.

    Returns:
        str: L'ID de la vidéo YouTube uploadée si succès, sinon None.
    """
    print(f"📤 Démarrage de l'upload YouTube pour : {video_path}")
    if not os.path.exists(video_path):
        print(f"❌ Erreur : Le fichier vidéo n'existe pas à {video_path}")
        return None

    # Assurez-vous que les tags sont une liste de chaînes
    tags_raw = metadata.get('tags', [])
    if isinstance(tags_raw, list):
        processed_tags = [str(tag).strip() for tag in tags_raw if str(tag).strip()]
    elif isinstance(tags_raw, str):
        processed_tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
    else:
        processed_tags = []

    body = {
        'snippet': {
            'title': metadata.get('title', 'Short Twitch'),
            'description': metadata.get('description', ''),
            'tags': processed_tags,
            'categoryId': str(metadata.get('categoryId', '20')),
            'defaultLanguage': 'fr',
            'defaultAudioLanguage': 'fr'
        },
        'status': {
            'privacyStatus': metadata.get('privacyStatus', 'public'),
            'embeddable': metadata.get('embeddable', True),
            'license': metadata.get('license', 'youtube'),
            'selfDeclaredMadeForKids': metadata.get('selfDeclaredMadeForKids', False)
        }
    }

    # Pour marquer comme Short, la vidéo doit être verticale (rapport 9:16) et <= 60s
    # Le code ne vérifie pas le rapport ici, il faut s'assurer que le clip source est bien vertical
    # ou que le traitement vidéo le convertit. YouTube le détecte automatiquement comme Short.

    media = MediaFileUpload(video_path, resumable=True)

    try:
        request = youtube_service.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media
        )
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"Progression de l'upload : {int(status.resumable_progress * 100)}%")
        
        video_id = response.get('id')
        print(f"✅ Vidéo uploadée avec succès ! ID de la vidéo : {video_id}")
        print(f"Lien : https://youtu.be/{video_id}")
        return video_id

    except HttpError as e:
        error_details = json.loads(e.content.decode('utf-8'))
        print(f"❌ Erreur lors de l'upload YouTube (HttpError) : {e}")
        print(f"Détails de l'erreur API : {error_details}")
        if 'error' in error_details and 'errors' in error_details['error']:
            for err in error_details['error']['errors']:
                print(f"  Raison: {err.get('reason')}")
                print(f"  Message: {err.get('message')}")
        return None
    except Exception as e:
        print(f"❌ Une erreur inattendue est survenue lors de l'upload : {e}")
        return None

if __name__ == "__main__":
    # Ce script est conçu pour être appelé par main.py
    print("Ce script est conçu pour être exécuté via main.py.")
    print("Pour une utilisation locale, assurez-vous que 'client_secret.json' est présent et configurez votre environnement.")
    
    # Pour le premier run local, vous devrez interagir pour l'authentification.
    # youtube = get_authenticated_service()
    
    # if youtube:
    #     print("Service YouTube prêt.")
    #     # Simulez des données de vidéo et un chemin de fichier
    #     # video_file_to_upload = os.path.join("data", "processed_clip_test.mp4") # Assurez-vous que ce fichier existe
    #     # if not os.path.exists(video_file_to_upload):
    #     #     print(f"Erreur: Le fichier '{video_file_to_upload}' n'existe pas pour le test d'upload.")
    #     # else:
    #     #     test_metadata = {
    #     #         "title": "Test Short par mon script Python",
    #     #         "description": "Ceci est un test d'upload de Short via un script Python.",
    #     #         "tags": ["test", "python", "youtube", "short"], # Doit être une liste ici pour le test
    #     #         "categoryId": "20", # Gaming
    #     #         "privacyStatus": "private", # Mettez en privé pour les tests
    #     #         "selfDeclaredMadeForKids": False,
    #     #         "embeddable": True,
    #     #         "license": "youtube"
    #     #     }
    #     #     uploaded_video_id = upload_youtube_short(youtube, video_file_to_upload, test_metadata)
    #     #     if uploaded_video_id:
    #     #         print(f"Test d'upload réussi. ID: {uploaded_video_id}")