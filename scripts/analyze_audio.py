# scripts/analyze_audio.py
"""
Analyse audio d'un clip : détection du segment le plus intense (hook)
et des pics sonores pour déclencher des effets visuels.

Utilise uniquement NumPy — pas de dépendance librosa.
"""

import numpy as np


def analyze_audio(video_path, hook_duration=3.0, peak_threshold_factor=1.4):
    """
    Analyse l'audio d'une vidéo pour trouver le hook (passage le plus fort)
    et les pics sonores exploitables pour des flashs.

    Args:
        video_path: chemin du fichier vidéo
        hook_duration: durée cible du hook en secondes (défaut 3s)
        peak_threshold_factor: multiplicateur d'écart-type pour seuil de pic

    Returns:
        dict: {"hook_start": float, "hook_end": float, "peaks": [float, ...]}
    """

    from moviepy.editor import VideoFileClip

    clip = VideoFileClip(video_path)
    duration = clip.duration

    if clip.audio is None:
        clip.close()
        return {
            "hook_start": 0.0,
            "hook_end": min(hook_duration, duration),
            "peaks": [],
        }

    try:
        # Extraction audio mono @ 22 kHz
        audio_arr = clip.audio.to_soundarray(fps=22050, nbytes=2, quantize=True)
        clip.close()

        if audio_arr.size == 0:
            return {
                "hook_start": 0.0,
                "hook_end": min(hook_duration, duration),
                "peaks": [],
            }

        if audio_arr.ndim > 1:
            audio_mono = audio_arr.mean(axis=1)
        else:
            audio_mono = audio_arr

        sample_rate = 22050
        total_samples = len(audio_mono)
        total_duration = max(total_samples / sample_rate, 0.1)

        # Fenêtres glissantes de ~0,08 s (pas de 0,04 s = 50 % de recouvrement)
        window_size = int(sample_rate * 0.08)
        step = window_size // 2
        rms_vals = []

        for start in range(0, total_samples - window_size, step):
            window = audio_mono[start : start + window_size]
            rms = float(np.sqrt(np.mean(window.astype(np.float64) ** 2)))
            rms_vals.append(rms)

        rms_vals = np.array(rms_vals, dtype=np.float64)

        if len(rms_vals) == 0:
            return {
                "hook_start": 0.0,
                "hook_end": min(hook_duration, total_duration),
                "peaks": [],
            }

        # ---- Hook : fenêtre de hook_duration avec la RMS moyenne la plus élevée ----
        windows_in_hook = int(hook_duration / (step / sample_rate))
        if windows_in_hook > len(rms_vals):
            windows_in_hook = len(rms_vals)

        best_start = 0
        best_energy = 0.0
        for i in range(len(rms_vals) - windows_in_hook + 1):
            energy = float(np.mean(rms_vals[i : i + windows_in_hook]))
            if energy > best_energy:
                best_energy = energy
                best_start = i

        hook_start = best_start * (step / sample_rate)
        hook_end = hook_start + hook_duration
        hook_start = max(0.0, min(hook_start, total_duration - hook_duration))
        hook_end = hook_start + hook_duration

        # ---- Pics (flash triggers) ----
        mean_rms = float(np.mean(rms_vals))
        std_rms = float(np.std(rms_vals))
        threshold = mean_rms + peak_threshold_factor * std_rms

        peaks = []
        min_gap_samples = int(0.8 / (step / sample_rate))  # au moins 0,8 s entre deux pics
        last_peak_idx = -min_gap_samples

        for i in range(1, len(rms_vals) - 1):
            if rms_vals[i] > threshold and rms_vals[i - 1] < rms_vals[i] > rms_vals[i + 1]:
                if i - last_peak_idx >= min_gap_samples:
                    peak_time = i * (step / sample_rate)
                    peaks.append(float(peak_time))
                    last_peak_idx = i

        # Garder les 15 plus forts
        if len(peaks) > 15:
            peak_energies = [float(rms_vals[int(p / (step / sample_rate))]) for p in peaks]
            idx_sorted = np.argsort(peak_energies)[::-1][:15]
            peaks = [peaks[j] for j in sorted(idx_sorted)]

        peaks.sort()

        print(
            f"🔊 Analyse audio OK — hook={hook_start:.1f}s–{hook_end:.1f}s, "
            f"{len(peaks)} pic(s)"
        )
        return {
            "hook_start": hook_start,
            "hook_end": hook_end,
            "peaks": peaks,
        }

    except Exception as exc:
        print(f"⚠️  Analyse audio impossible : {exc}")
        try:
            clip.close()
        except Exception:
            pass
        return {
            "hook_start": 0.0,
            "hook_end": min(hook_duration, duration),
            "peaks": [],
        }