# scripts/analyze_audio.py
"""
Analyse audio d'un clip : detection du segment le plus intense (hook)
et des pics sonores pour declencher des effets visuels.

Extraction audio via ffmpeg (fiable) puis calcul RMS avec NumPy.
"""

import os
import subprocess
import tempfile
import wave

import numpy as np


def _extract_mono_audio(video_path, sample_rate=22050):
    """
    Extrait la piste audio en WAV mono via ffmpeg.
    Retourne (np.array float64 [-1,1], duration_seconds) ou (None, 0).
    """
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None, 0.0

    wav_path = os.path.join(
        tempfile.gettempdir(), f"_audio_{os.getpid()}.wav"
    )

    cmd = [
        ffmpeg_exe, "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "wav", wav_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception as exc:
        print(f"ffmpeg audio extraction echouee : {exc}")
        return None, 0.0

    try:
        with wave.open(wav_path, "rb") as wf:
            n_frames = wf.getnframes()
            data = wf.readframes(n_frames)
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float64) / 32768.0
        duration = len(audio) / sample_rate
        return audio, duration
    except Exception as exc:
        print(f"Lecture WAV echouee : {exc}")
        return None, 0.0
    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass


def analyze_audio(video_path, hook_duration=3.0, peak_threshold_factor=1.4):
    """
    Analyse l'audio d'une video pour trouver le hook (passage le plus fort)
    et les pics sonores exploitables pour des flashs.

    Args:
        video_path: chemin du fichier video
        hook_duration: duree cible du hook en secondes (defaut 3s)
        peak_threshold_factor: multiplicateur d'ecart-type pour seuil de pic

    Returns:
        dict: {"hook_start": float, "hook_end": float, "peaks": [float, ...]}
    """
    sample_rate = 22050

    # Duree de reference via moviepy (leger) pour les valeurs par defaut
    fallback_duration = 60.0
    try:
        from moviepy.editor import VideoFileClip
        with VideoFileClip(video_path) as c:
            fallback_duration = c.duration
    except Exception:
        pass

    empty_result = {
        "hook_start": 0.0,
        "hook_end": min(hook_duration, fallback_duration),
        "peaks": [],
    }

    audio_mono, total_duration = _extract_mono_audio(video_path, sample_rate)

    if audio_mono is None or len(audio_mono) == 0:
        print("Audio indisponible - hook par defaut au debut du clip.")
        return empty_result

    # ---- Fenetres glissantes 0.08s, pas 0.04s ----
    window_size = int(sample_rate * 0.08)
    step = window_size // 2
    rms_vals = []

    for start in range(0, len(audio_mono) - window_size, step):
        window = audio_mono[start : start + window_size]
        rms = float(np.sqrt(np.mean(window ** 2)))
        rms_vals.append(rms)

    rms_vals = np.array(rms_vals, dtype=np.float64)

    if len(rms_vals) == 0:
        return empty_result

    # ---- Hook : fenetre de hook_duration avec RMS moyenne max ----
    windows_in_hook = int(hook_duration / (step / sample_rate))
    windows_in_hook = min(windows_in_hook, len(rms_vals))

    best_start = 0
    best_energy = 0.0
    for i in range(len(rms_vals) - windows_in_hook + 1):
        energy = float(np.mean(rms_vals[i : i + windows_in_hook]))
        if energy > best_energy:
            best_energy = energy
            best_start = i

    hook_start = best_start * (step / sample_rate)
    hook_start = max(0.0, min(hook_start, total_duration - hook_duration))
    hook_end = hook_start + hook_duration

    # ---- Pics (flash triggers) ----
    mean_rms = float(np.mean(rms_vals))
    std_rms = float(np.std(rms_vals))
    threshold = mean_rms + peak_threshold_factor * std_rms

    peaks = []
    min_gap_samples = int(0.8 / (step / sample_rate))
    last_peak_idx = -min_gap_samples

    for i in range(1, len(rms_vals) - 1):
        if rms_vals[i] > threshold and rms_vals[i - 1] < rms_vals[i] > rms_vals[i + 1]:
            if i - last_peak_idx >= min_gap_samples:
                peaks.append(float(i * (step / sample_rate)))
                last_peak_idx = i

    # Garder les 15 plus forts
    if len(peaks) > 15:
        energies = [float(rms_vals[int(p / (step / sample_rate))]) for p in peaks]
        idx_sorted = np.argsort(energies)[::-1][:15]
        peaks = [peaks[j] for j in sorted(idx_sorted)]

    peaks.sort()

    print(
        f"Analyse audio OK - hook={hook_start:.1f}s-{hook_end:.1f}s, "
        f"{len(peaks)} pic(s)"
    )
    return {
        "hook_start": hook_start,
        "hook_end": hook_end,
        "peaks": peaks,
    }