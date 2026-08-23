# scripts/detect_webcam.py
"""
Detection robuste de la zone webcam.
Essaie MediaPipe, puis fallback OpenCV Haar cascades.
Echantillonnage multi-frames (10 frames) + verification de coherence spatiale.
"""

import os
import numpy as np


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def _find_or_download_cascade():
    """Trouve ou telecharge le fichier cascade Haar pour OpenCV."""
    import cv2

    # Chercher dans les chemins standards
    candidates = [
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"),
    ]
    # cv2 peut etre dans un dossier data/ a cote du module
    try:
        cv2_dir = os.path.dirname(cv2.__file__)
        candidates.append(os.path.join(cv2_dir, "data", "haarcascade_frontalface_default.xml"))
    except Exception:
        pass

    # Chercher recursivement dans site-packages
    try:
        import site
        for sp in site.getsitepackages():
            for root, dirs, files in os.walk(sp):
                if "haarcascade_frontalface_default.xml" in files:
                    candidates.append(os.path.join(root, "haarcascade_frontalface_default.xml"))
                    break
    except Exception:
        pass

    for path in candidates:
        if os.path.exists(path):
            return path

    # Dernier recours : telecharger
    cascade_url = (
        "https://raw.githubusercontent.com/opencv/opencv/master/data/"
        "haarcascades/haarcascade_frontalface_default.xml"
    )
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "opencv_cascade")
    local_path = os.path.join(cache_dir, "haarcascade_frontalface_default.xml")

    if os.path.exists(local_path):
        return local_path

    try:
        import requests
        os.makedirs(cache_dir, exist_ok=True)
        r = requests.get(cascade_url, timeout=15)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
        print(f"Cascade Haar telechargee dans {local_path}")
        return local_path
    except Exception as e:
        print(f"Impossible de telecharger la cascade Haar : {e}")
        return None


def _get_detector(confidence_threshold=0.65):
    """
    Essaie plusieurs backends pour la detection de visage.
    Retourne (detector, backend_name) ou (None, None).
    """
    errors = []

    # 1. mediapipe.solutions (API legacy)
    try:
        import mediapipe as mp
        mp_face = mp.solutions.face_detection
        detector = mp_face.FaceDetection(
            model_selection=0, min_detection_confidence=confidence_threshold
        )
        return detector, "mediapipe.solutions"
    except Exception as e:
        errors.append(f"mediapipe.solutions: {e}")

    # 2. mediapipe.python.solutions
    try:
        from mediapipe.python.solutions import face_detection as mp_face
        detector = mp_face.FaceDetection(
            model_selection=0, min_detection_confidence=confidence_threshold
        )
        return detector, "mediapipe.python.solutions"
    except Exception as e:
        errors.append(f"mediapipe.python.solutions: {e}")

    # 3. Fallback OpenCV Haar cascade (local)
    try:
        import cv2
        cascade_path = _find_or_download_cascade()
        if cascade_path:
            detector = cv2.CascadeClassifier(cascade_path)
            return detector, "opencv_haar"
        else:
            errors.append("opencv: cascade introuvable")
    except Exception as e:
        errors.append(f"opencv: {e}")

    print(f"Erreurs de chargement des backends: {' | '.join(errors)}")
    return None, None


def _detect_faces_mediapipe(detector, frame):
    """Retourne une liste de bboxes (x1,y1,x2,y2) via MediaPipe."""
    results = detector.process(frame)
    bboxes = []
    if results.detections:
        h, w = frame.shape[:2]
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)
            bboxes.append((x1, y1, x2, y2))
    return bboxes


def _detect_faces_opencv(detector, frame):
    """Retourne une liste de bboxes (x1,y1,x2,y2) via OpenCV Haar."""
    gray = frame
    if gray.ndim == 3:
        try:
            import cv2
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        except Exception:
            return []
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(40, 40)
    )
    return [(x, y, x + w, y + h) for (x, y, w, h) in faces]


# ---------------------------------------------------------------------------
# API principale
# ---------------------------------------------------------------------------


def detect_webcam(video_path, num_samples=10, confidence_threshold=0.65):
    """
    Analyse plusieurs frames du clip pour detecter une webcam (visage du streamer).

    Retourne :
        dict : {"has_webcam": bool, "bbox": {...}, "sample_positions": [...]}
    """
    from moviepy.editor import VideoFileClip

    detector, backend = _get_detector(confidence_threshold)
    if detector is None:
        print("Aucun backend de detection de visage disponible (mediapipe / opencv).")
        return _no_webcam()

    print(f"Detection webcam : backend = {backend}")

    if backend == "mediapipe":
        detect_fn = _detect_faces_mediapipe
    else:
        detect_fn = _detect_faces_opencv

    clip = VideoFileClip(video_path)
    duration = clip.duration
    w, h = clip.size

    # Points d'echantillonnage
    sample_times = []
    segment = max(duration / (num_samples + 1), 0.5)
    for i in range(1, num_samples + 1):
        t = min(i * segment, duration - 0.1)
        if t > 0:
            sample_times.append(t)

    if not sample_times:
        clip.close()
        return _no_webcam()

    detected_bboxes = []
    sample_positions = []

    for t in sample_times:
        try:
            frame = clip.get_frame(t)
        except Exception:
            continue

        try:
            bboxes = detect_fn(detector, frame)
        except Exception:
            bboxes = []

        if bboxes:
            # Prendre le plus grand visage
            best = max(bboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            detected_bboxes.append(best)
            sample_positions.append({"t": t, "bbox": best, "score": 1.0})
        else:
            sample_positions.append({"t": t, "bbox": None, "score": 0.0})

    clip.close()

    if backend == "mediapipe" and hasattr(detector, "close"):
        detector.close()

    hit_rate = len(detected_bboxes) / max(len(sample_times), 1)

    if hit_rate < 0.6 or len(detected_bboxes) < 2:
        print(
            f"Webcam non detectee (hit rate {hit_rate:.0%}, "
            f"{len(detected_bboxes)}/{len(sample_times)}) -> plein ecran"
        )
        return _no_webcam(sample_positions)

    # Coherence spatiale
    boxes = np.array(detected_bboxes, dtype=np.float64)
    centers = np.column_stack(
        [(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]
    )
    center_std = np.std(centers, axis=0)
    avg_box = tuple(int(v) for v in np.mean(boxes, axis=0))

    if center_std[0] > w * 0.25 or center_std[1] > h * 0.25:
        print("Visages detectes mais positions trop variables -> plein ecran")
        return _no_webcam(sample_positions)

    # Marge 15%
    x1, y1, x2, y2 = avg_box
    margin_x = int((x2 - x1) * 0.15)
    margin_y = int((y2 - y1) * 0.15)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    print(
        f"Webcam detectee (hit rate {hit_rate:.0%}) -> bbox=({x1},{y1},{x2},{y2})"
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