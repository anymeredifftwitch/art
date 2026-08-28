# scripts/generate_subtitles.py
"""
Transcription automatique français via faster-whisper (tiny).
Retourne des groupes de 2-4 mots avec timestamps pour affichage style Shorts.
"""

import os
import tempfile


def _extract_audio(video_path, audio_path):
    """Extrait la piste audio d'une vidéo au format WAV 16 kHz mono."""
    from moviepy.editor import VideoFileClip

    clip = VideoFileClip(video_path)
    if clip.audio is None:
        clip.close()
        return None

    clip.audio.write_audiofile(
        audio_path, fps=16000, codec="pcm_s16le", logger=None
    )
    clip.close()
    return audio_path


def transcribe(video_path, model_size="medium", device="cpu", compute_type="int8"):
    """
    Transcrit la parole du clip.

    Retourne :
        list[dict] : [{"text": "mot1 mot2", "start": float, "end": float}, ...]
        Groupes de 2-4 mots.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("❌ faster-whisper non installé. Ajoute-le dans requirements.txt.")
        return []

    audio_path = os.path.join(
        tempfile.gettempdir(), f"_subtitles_audio_{os.getpid()}.wav"
    )

    if _extract_audio(video_path, audio_path) is None:
        return []

    try:
        print("🧠 Transcription en cours (Whisper medium)…")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments, _ = model.transcribe(
            audio_path,
            language="fr",
            word_timestamps=True,
            vad_filter=True,
            beam_size=5,
        )

        words = []
        for seg in segments:
            if seg.words is None:
                continue
            for w in seg.words:
                # w.start / w.end sont des secondes
                words.append(
                    {
                        "word": w.word.strip() if w.word.strip() else ".",
                        "start": w.start,
                        "end": w.end,
                    }
                )

        # Grouper par paquets de 2-4 mots
        groups = []
        GROUP_SIZE = 3
        for i in range(0, len(words), GROUP_SIZE):
            group = words[i : i + GROUP_SIZE]
            text = " ".join(w["word"] for w in group)
            start = group[0]["start"]
            end = group[-1]["end"]
            groups.append({
                "text": text,
                "start": start,
                "end": end,
                "words": group,
            })

        print(f"✅ Transcription OK — {len(words)} mots, {len(groups)} groupes")
        return groups

    except Exception as exc:
        print(f"⚠️  Transcription échouée ({exc}) — sous-titres désactivés.")
        return []

    finally:
        try:
            os.remove(audio_path)
        except Exception:
            pass