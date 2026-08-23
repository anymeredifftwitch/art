# scripts/detect_webcam.py
"""
Détection robuste de la zone webcam via MediaPipe Face Detection.
Échantillonnage multi-frames (10 frames) + vérification de cohérence spatiale.
"""

import os
import numpy as np


def detect_webcam(video_path, num_samples=10, confidence_threshold=0.65):
    """
    Analyse plusieurs frames du clip pour détecter une webcam (visage du streamer).

    Retourne :
        dict | None :
            {
                "has_webcam": bool,
                "bbox": {"x1": int, "y1": int, "x2": int, "y2": int},
                "sample_positions": list[dict],  # pour debug / tracking
            }
    """
    try:
        import mediapipe as mp
    except ImportError:
        print("❌ mediapipe non installé. Détection webcam désactivée.")
        return _no_webcam()

    from moviepy.editor import VideoFileClip

    clip = VideoFileClip(video_path)
    duration = clip.duration
    w, h = clip.size

    # Points d'échantillonnage répartis sur toute la durée
    sample_times = []
    segment = max(duration / (num_samples + 1), 0.5)
    for i in range(1, num_samples + 1):
        t = min(i * segment, duration - 0.1)
        if t > 0:
            sample_times.append(t)

    if not sample_times:
        clip.close()
        return _no_webcam()

    mp_face = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection(
        model_selection=0, min_detection_confidence=confidence_threshold
    )

    detected_bboxes = []
    sample_positions = []

    for t in sample_times:
        try:
            frame = clip.get_frame(t)
        except Exception:
            continue

        frame_h, frame_w = frame.shape[:2]
        results = face_detector.process(frame)

        if results.detections:
            best = results.detections[0]
            bbox = best.location_data.relative_bounding_box
            x1 = int(bbox.xmin * frame_w)
            y1 = int(bbox.ymin * frame_h)
            x2 = int((bbox.xmin + bbox.width) * frame_w)
            y2 = int((bbox.ymin + bbox.height) * frame_h)
            detected_bboxes.append((x1, y1, x2, y2))
            sample_positions.append(
                {"t": t, "bbox": (x1, y1, x2, y2), "score": best.score[0]}
            )
        else:
            sample_positions.append({"t": t, "bbox": None, "score": 0.0})

    clip.close()
    face_detector.close()

    hit_rate = len(detected_bboxes) / len(sample_times)

    # Il faut au moins 60 % de frames avec visage
    if hit_rate < 0.6 or len(detected_bboxes) < 2:
        print(
            f"🔍 Webcam non détectée (hit rate {hit_rate:.0%}, "
            f"{len(detected_bboxes)}/{len(sample_times)}) → plein écran"
        )
        return _no_webcam(sample_positions)

    # Vérifier la cohérence spatiale (variance)
    boxes = np.array(detected_bboxes, dtype=np.float64)
    centers = np.column_stack(
        [(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]
    )
    center_std = np.std(centers, axis=0)
    avg_box = tuple(int(v) for v in np.mean(boxes, axis=0))

    if center_std[0] > w * 0.25 or center_std[1] > h * 0.25:
        print(
            "🔍 Visages détectés mais positions trop variables → plein écran"
        )
        return _no_webcam(sample_positions)

    # Marge de 15 %
    x1, y1, x2, y2 = avg_box
    margin_x = int((x2 - x1) * 0.15)
    margin_y = int((y2 - y1) * 0.15)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    print(
        f"✅ Webcam détectée (hit rate {hit_rate:.0%}) — bbox=({x1},{y1},{x2},{y2})"
    )
    return {
        "has_webcam": True,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "sample_positions": sample_positions,
    }


def _no_webcam(positions=None):
    return {
        "has_webcam": False,
        "bbox": None,
        "sample_positions": positions or [],
    }