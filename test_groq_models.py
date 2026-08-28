#!/usr/bin/env python3
"""
Diagnostic et Benchmark des Modèles Groq API
=============================================
Ce script liste tous les modèles disponibles sur votre compte Groq,
les teste un par un avec un prompt réel et affiche un tableau comparatif
(Statut, Temps de réponse, Exemple de génération, Erreurs).

Usage :
    python test_groq_models.py
    python test_groq_models.py --api-key gsk_...
"""

import os
import sys
import time
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_api_key(cli_arg_key=None):
    """Récupère la clé API via argument, variable d'environnement ou saisie."""
    key = cli_arg_key or os.getenv("GROQ_API_KEY")
    if not key:
        print("\n🔑 Aucune clé GROQ_API_KEY trouvée dans l'environnement / .env")
        try:
            key = input("👉 Entrez votre clé API Groq (gsk_...) : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAnnulation.")
            sys.exit(1)
    return key


def main():
    parser = argparse.ArgumentParser(description="Test et benchmark de tous les modèles Groq")
    parser.add_argument("--api-key", type=str, default=None, help="Clé API Groq")
    args = parser.parse_args()

    api_key = get_api_key(args.api_key)
    if not api_key:
        print("❌ Clé API manquante. Arrêt.")
        sys.exit(1)

    try:
        from groq import Groq
    except ImportError:
        print("❌ La bibliothèque 'groq' n'est pas installée.")
        print("Installez-la avec : pip install groq")
        sys.exit(1)

    print(f"\n📡 Connexion à l'API Groq avec la clé : {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else ''}")
    client = Groq(api_key=api_key)

    # 1. Récupération des modèles via l'API
    print("🔍 Récupération de la liste des modèles officiels...")
    try:
        api_models_resp = client.models.list()
        api_model_ids = [m.id for m in api_models_resp.data]
        print(f"✅ {len(api_model_ids)} modèles trouvés sur l'API Groq.")
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des modèles : {e}")
        api_model_ids = []

    # Modèles connus à tester impérativement (fusionnés avec ceux de l'API)
    known_models = [
        "llama-3.3-70b-versatile",
        "llama-3.3-70b-specdec",
        "llama-3.1-8b-instant",
        "deepseek-r1-distill-llama-70b",
        "deepseek-r1-distill-qwen-32b",
        "qwen-2.5-32b",
        "qwen-2.5-coder-32b",
        "gemma2-9b-it",
        "llama-3.2-11b-vision-preview",
        "llama-3.2-3b-preview",
        "llama-3.2-1b-preview",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
    ]

    all_models = list(dict.fromkeys(known_models + api_model_ids))
    # Exclure les modèles de transcription / audio / embeddings du test chat
    chat_models = [
        m for m in all_models
        if not any(x in m.lower() for x in ["whisper", "embed", "tts", "guard", "audio"])
    ]

    print(f"🧪 {len(chat_models)} modèles de chat/texte vont être testés.\n")
    print("=" * 90)
    print(f"{'MODÈLE':<35} | {'STATUT':<10} | {'TEMPS (ms)':<10} | {'RÉPONSE / ERREUR'}")
    print("=" * 90)

    test_prompt = "Génère un titre Twitch court et percutant de 4 mots pour un streamer qui fail en direct."

    working_models = []
    failed_models = []

    for model in chat_models:
        start_t = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Tu réponds uniquement le titre court, sans guillemets ni introduction."},
                    {"role": "user", "content": test_prompt},
                ],
                temperature=0.7,
                max_tokens=40,
            )
            elapsed_ms = int((time.time() - start_t) * 1000)
            text = resp.choices[0].message.content.strip().replace("\n", " ")
            if len(text) > 35:
                text = text[:32] + "..."
            print(f"{model:<35} | {'✅ OK':<10} | {f'{elapsed_ms} ms':<10} | {text}")
            working_models.append((model, elapsed_ms, text))
        except Exception as e:
            elapsed_ms = int((time.time() - start_t) * 1000)
            err_msg = str(e)
            if "model_decommissioned" in err_msg or "decommissioned" in err_msg:
                short_err = "❌ DÉCOMMISSIONNÉ"
            elif "model_not_found" in err_msg or "does not exist" in err_msg:
                short_err = "❌ NON TROUVÉ"
            elif "rate_limit" in err_msg:
                short_err = "⚠️ RATE LIMIT"
            else:
                short_err = f"❌ {err_msg[:35]}..."
            print(f"{model:<35} | {'❌ FAIL':<10} | {f'{elapsed_ms} ms':<10} | {short_err}")
            failed_models.append((model, err_msg))

    print("=" * 90)
    print(f"\n📊 BILAN : {len(working_models)} fonctionnels / {len(failed_models)} échoués")

    if working_models:
        print("\n✨ MODÈLES OPÉRATIONNELS VALIDÉS :")
        for m, ms, sample in working_models:
            print(f"  • {m:<32} (latence: {ms}ms) -> « {sample} »")

        best = working_models[0][0]
        print(f"\n🏆 Modèle recommandé en priorité 1 : {best}")
    else:
        print("\n⚠️ Aucun modèle n'a fonctionné. Vérifiez la validité de votre clé API ou les restrictions de votre compte.")


if __name__ == "__main__":
    main()
