import os
import cv2
import time
import math
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

from collections import Counter, deque
from tensorflow.keras.models import load_model
from joblib import load
from ultralytics import YOLO
BASE_DIR = os.path.abspath("..")

EYE_MODEL_PATH = os.path.join(BASE_DIR, "notebooks", "best_eye_model_v3.keras")
YAWN_MODEL_PATH = os.path.join(BASE_DIR, "notebooks", "best_yawn_landmark_model.pkl")
YAWN_IMPUTER_PATH = os.path.join(BASE_DIR, "notebooks", "yawn_imputer.pkl")

# Phone object detector
PHONE_MODEL_PATH = "yolov8n.pt"


print("Eye model exists:", os.path.exists(EYE_MODEL_PATH), EYE_MODEL_PATH)
print("Yawn model exists:", os.path.exists(YAWN_MODEL_PATH), YAWN_MODEL_PATH)
print("Yawn imputer exists:", os.path.exists(YAWN_IMPUTER_PATH), YAWN_IMPUTER_PATH)
eye_model = load_model(EYE_MODEL_PATH, compile=False)
yawn_model = load(YAWN_MODEL_PATH)
yawn_imputer = load(YAWN_IMPUTER_PATH)

phone_model = YOLO(PHONE_MODEL_PATH)

print("All models loaded successfully.")
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("MediaPipe FaceMesh initialized.")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

IMG_SIZE = 224
USE_GRAYSCALE = False
CLOSED_THRESHOLD = 0.50
STABLE_FRAMES = 3
MAX_MISSING_EYE_FRAMES = 5

print("Haar cascades loaded.")
def clamp(v, low, high):
    return max(low, min(v, high))


def crop_from_landmarks(frame, pts, pad=8):
    h, w, _ = frame.shape

    xs = [int(p[0]) for p in pts]
    ys = [int(p[1]) for p in pts]

    x1 = clamp(min(xs) - pad, 0, w - 1)
    y1 = clamp(min(ys) - pad, 0, h - 1)
    x2 = clamp(max(xs) + pad, 0, w - 1)
    y2 = clamp(max(ys) + pad, 0, h - 1)

    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1, x2, y2)

    crop = frame[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


def get_face_mesh_points(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None, None

    h, w, _ = frame.shape
    face_landmarks = results.multi_face_landmarks[0]

    points = []
    for lm in face_landmarks.landmark:
        points.append((lm.x * w, lm.y * h))

    return points, face_landmarks
def preprocess_eye(eye_img, img_size=224, use_grayscale=False):
    if eye_img is None or eye_img.size == 0:
        return None

    if use_grayscale:
        eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        eye = cv2.resize(eye, (img_size, img_size))
        eye = eye.astype("float32") / 255.0
        eye = np.expand_dims(eye, axis=-1)
        eye = np.expand_dims(eye, axis=0)
    else:
        eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
        eye = cv2.resize(eye, (img_size, img_size))
        eye = eye.astype("float32") / 255.0
        eye = np.expand_dims(eye, axis=0)

    return eye


def decode_eye_prediction(pred, closed_threshold=0.50):
    pred = np.array(pred)

    if pred.shape[-1] == 1:
        closed_prob = float(pred[0][0])
        open_prob = 1.0 - closed_prob
        state = "closed" if closed_prob >= closed_threshold else "open"
        return state, open_prob, closed_prob

    elif pred.shape[-1] == 2:
        open_prob = float(pred[0][0])
        closed_prob = float(pred[0][1])
        state = "closed" if closed_prob >= closed_threshold else "open"
        return state, open_prob, closed_prob

    else:
        raise ValueError(f"Unsupported eye model output shape: {pred.shape}")


LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]
def detect_eyes(frame):
    annotated = frame.copy()

    points, face_landmarks = get_face_mesh_points(frame)
    if points is None:
        return {
            "eyes_label": "unknown",
            "eyes_closed": False,
            "confidence": 0.0,
            "open_prob": 0.0,
            "closed_prob": 0.0,
            "eyes_found": 0,
            "annotated": annotated
        }

    left_eye_pts = [points[i] for i in LEFT_EYE_IDX]
    right_eye_pts = [points[i] for i in RIGHT_EYE_IDX]

    left_crop, (lx1, ly1, lx2, ly2) = crop_from_landmarks(frame, left_eye_pts, pad=8)
    right_crop, (rx1, ry1, rx2, ry2) = crop_from_landmarks(frame, right_eye_pts, pad=8)

    open_probs = []
    closed_probs = []
    eyes_found = 0

    # draw eye boxes
    cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2)
    cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)

    # left eye
    if left_crop is not None and left_crop.size > 0:
        left_input = preprocess_eye(left_crop, IMG_SIZE, USE_GRAYSCALE)
        if left_input is not None:
            pred = eye_model.predict(left_input, verbose=0)
            _, open_prob, closed_prob = decode_eye_prediction(pred, CLOSED_THRESHOLD)
            open_probs.append(open_prob)
            closed_probs.append(closed_prob)
            eyes_found += 1

    # right eye
    if right_crop is not None and right_crop.size > 0:
        right_input = preprocess_eye(right_crop, IMG_SIZE, USE_GRAYSCALE)
        if right_input is not None:
            pred = eye_model.predict(right_input, verbose=0)
            _, open_prob, closed_prob = decode_eye_prediction(pred, CLOSED_THRESHOLD)
            open_probs.append(open_prob)
            closed_probs.append(closed_prob)
            eyes_found += 1

    if eyes_found == 0:
        return {
            "eyes_label": "unknown",
            "eyes_closed": False,
            "confidence": 0.0,
            "open_prob": 0.0,
            "closed_prob": 0.0,
            "eyes_found": 0,
            "annotated": annotated
        }

    avg_open_prob = float(np.mean(open_probs))
    avg_closed_prob = float(np.mean(closed_probs))

    final_state = "closed" if avg_closed_prob >= CLOSED_THRESHOLD else "open"

    return {
        "eyes_label": final_state,
        "eyes_closed": final_state == "closed",
        "confidence": avg_closed_prob,
        "open_prob": avg_open_prob,
        "closed_prob": avg_closed_prob,
        "eyes_found": eyes_found,
        "annotated": annotated
    }
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

UPPER_INNER = 13
LOWER_INNER = 14

UPPER_OUTER = 0
LOWER_OUTER = 17

UPPER_MID = 12
LOWER_MID = 15

LEFT_UPPER = 78
RIGHT_UPPER = 308
LEFT_LOWER = 95
RIGHT_LOWER = 324


def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def extract_landmarks_from_image(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None

    h, w, _ = image_bgr.shape
    face_landmarks = results.multi_face_landmarks[0]

    points = []
    for lm in face_landmarks.landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

    return points


def compute_mouth_features(points):
    left_mouth = points[LEFT_MOUTH]
    right_mouth = points[RIGHT_MOUTH]

    upper_inner = points[UPPER_INNER]
    lower_inner = points[LOWER_INNER]

    upper_outer = points[UPPER_OUTER]
    lower_outer = points[LOWER_OUTER]

    upper_mid = points[UPPER_MID]
    lower_mid = points[LOWER_MID]

    left_upper = points[LEFT_UPPER]
    right_upper = points[RIGHT_UPPER]
    left_lower = points[LEFT_LOWER]
    right_lower = points[RIGHT_LOWER]

    mouth_width = euclidean(left_mouth, right_mouth)
    inner_height = euclidean(upper_inner, lower_inner)
    outer_height = euclidean(upper_outer, lower_outer)
    mid_height = euclidean(upper_mid, lower_mid)

    left_opening = euclidean(left_upper, left_lower)
    right_opening = euclidean(right_upper, right_lower)

    if mouth_width < 1e-6:
        mouth_width = 1e-6

    mar_inner = inner_height / mouth_width
    mar_outer = outer_height / mouth_width
    mar_mid = mid_height / mouth_width
    mar_avg_side = ((left_opening + right_opening) / 2.0) / mouth_width

    features = [
        mouth_width,
        inner_height,
        outer_height,
        mid_height,
        left_opening,
        right_opening,
        mar_inner,
        mar_outer,
        mar_mid,
        mar_avg_side
    ]

    return np.array(features, dtype=np.float32)


def detect_yawning_raw(frame):
    points = extract_landmarks_from_image(frame)
    if points is None:
        return None, 0.0, None

    features = compute_mouth_features(points).reshape(1, -1)
    features = yawn_imputer.transform(features)

    pred_label = yawn_model.predict(features)[0]
    pred_prob = yawn_model.predict_proba(features)[0]
    yawn_prob = float(pred_prob[1])

    return pred_label, yawn_prob, points
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

UPPER_INNER = 13
LOWER_INNER = 14

UPPER_OUTER = 0
LOWER_OUTER = 17

UPPER_MID = 12
LOWER_MID = 15

LEFT_UPPER = 78
RIGHT_UPPER = 308
LEFT_LOWER = 95
RIGHT_LOWER = 324


def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def extract_landmarks_from_image(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None

    h, w, _ = image_bgr.shape
    face_landmarks = results.multi_face_landmarks[0]

    points = []
    for lm in face_landmarks.landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

    return points


def compute_mouth_features(points):
    left_mouth = points[LEFT_MOUTH]
    right_mouth = points[RIGHT_MOUTH]

    upper_inner = points[UPPER_INNER]
    lower_inner = points[LOWER_INNER]

    upper_outer = points[UPPER_OUTER]
    lower_outer = points[LOWER_OUTER]

    upper_mid = points[UPPER_MID]
    lower_mid = points[LOWER_MID]

    left_upper = points[LEFT_UPPER]
    right_upper = points[RIGHT_UPPER]
    left_lower = points[LEFT_LOWER]
    right_lower = points[RIGHT_LOWER]

    mouth_width = euclidean(left_mouth, right_mouth)
    inner_height = euclidean(upper_inner, lower_inner)
    outer_height = euclidean(upper_outer, lower_outer)
    mid_height = euclidean(upper_mid, lower_mid)

    left_opening = euclidean(left_upper, left_lower)
    right_opening = euclidean(right_upper, right_lower)

    if mouth_width < 1e-6:
        mouth_width = 1e-6

    mar_inner = inner_height / mouth_width
    mar_outer = outer_height / mouth_width
    mar_mid = mid_height / mouth_width
    mar_avg_side = ((left_opening + right_opening) / 2.0) / mouth_width

    features = [
        mouth_width,
        inner_height,
        outer_height,
        mid_height,
        left_opening,
        right_opening,
        mar_inner,
        mar_outer,
        mar_mid,
        mar_avg_side
    ]

    return np.array(features, dtype=np.float32)


def detect_yawning_raw(frame):
    points = extract_landmarks_from_image(frame)
    if points is None:
        return None, 0.0, None

    features = compute_mouth_features(points).reshape(1, -1)
    features = yawn_imputer.transform(features)

    pred_label = yawn_model.predict(features)[0]
    pred_prob = yawn_model.predict_proba(features)[0]
    yawn_prob = float(pred_prob[1])

    return pred_label, yawn_prob, points
def detect_phone_object(frame, conf_threshold=0.35):
    results = phone_model(
    frame,
    imgsz=480,
    conf=0.30,
    verbose=False
)

    phone_boxes = []
    best_conf = 0.0

    if len(results) == 0:
        return {"phone": False, "confidence": 0.0, "boxes": []}

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return {"phone": False, "confidence": 0.0, "boxes": []}

    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = phone_model.names[cls_id]

        if class_name == "cell phone" and conf >= conf_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            phone_boxes.append((x1, y1, x2, y2))
            best_conf = max(best_conf, conf)

    return {
        "phone": len(phone_boxes) > 0,
        "confidence": best_conf,
        "boxes": phone_boxes
    }


def smooth_class(class_history, new_class, window=8):
    class_history.append(new_class)
    if len(class_history) > window:
        class_history.pop(0)
    return Counter(class_history).most_common(1)[0][0]


def raw_class_to_state(raw_class):
    if raw_class in PHONE_CLASSES:
        return "PHONE"
    elif raw_class in TEXTING_CLASSES:
        return "TEXTING"
    elif raw_class in DISTRACTED_CLASSES:
        return "DISTRACTED"
    else:
        return "SAFE"

def detect_head_pose(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return {"direction": "unknown", "valid": False}

    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = frame.shape

    nose = face_landmarks.landmark[1]
    left_cheek = face_landmarks.landmark[234]
    right_cheek = face_landmarks.landmark[454]
    chin = face_landmarks.landmark[152]
    forehead = face_landmarks.landmark[10]

    nose_x = nose.x * w
    left_x = left_cheek.x * w
    right_x = right_cheek.x * w

    nose_y = nose.y * h
    chin_y = chin.y * h
    forehead_y = forehead.y * h

    face_center_x = (left_x + right_x) / 2.0
    face_center_y = (chin_y + forehead_y) / 2.0

    dx = nose_x - face_center_x
    dy = nose_y - face_center_y

    if dx > 45:
        direction = "left"
    elif dx < -45:
        direction = "right"
    elif dy > 40:
        direction = "down"
    elif dy < -30:
        direction = "up"
    else:
        direction = "forward"

    return {"direction": direction, "valid": True}
def compute_eyes_score(eyes_closed_duration):
    score = 0.0
    if eyes_closed_duration > 0.5:
        score = 0.3
    if eyes_closed_duration > 1.0:
        score = 0.6
    if eyes_closed_duration > 1.5:
        score = 0.8
    if eyes_closed_duration > 2.5:
        score = 1.0
    return min(score, 1.0)


def compute_yawn_score(yawn_duration, yawn_count):
    score = 0.0
    if yawn_duration > 0.7:
        score += 0.3
    if yawn_duration > 1.0:
        score += 0.3
    if yawn_duration > 1.5:
        score += 0.2
    if yawn_count >= 2:
        score += 0.1
    if yawn_count >= 4:
        score += 0.1
    return min(score, 1.0)


def compute_phone_score(phone_duration):
    score = 0.0
    if phone_duration > 0.5:
        score = 0.3
    if phone_duration > 1.0:
        score = 0.6
    if phone_duration > 2.0:
        score = 0.8
    if phone_duration > 3.0:
        score = 1.0
    return min(score, 1.0)


def compute_distraction_score(state, distraction_duration, phone_related_duration):
    score = 0.0

    if state == "PHONE":
        score += 0.6
    elif state == "TEXTING":
        score += 0.7
    elif state == "DISTRACTED":
        score += 0.4
    else:
        score += 0.1

    if distraction_duration > 1.0:
        score += 0.2
    if distraction_duration > 2.0:
        score += 0.2

    if phone_related_duration > 1.0:
        score += 0.2
    if phone_related_duration > 2.0:
        score += 0.1

    return min(score, 1.0)


def compute_head_score(head_direction, look_away_duration):
    if head_direction in ["left", "right", "down"]:
        if look_away_duration > 2.5:
            return 1.0
        elif look_away_duration > 1.5:
            return 0.7
        elif look_away_duration > 0.7:
            return 0.4
    return 0.0


def compute_final_risk(eyes_score, yawn_score, phone_score, head_score,
                       phone_duration, look_away_duration, eyes_closed_duration,
                       yawn_duration, yawn_count):
    # weighted score
    risk = (
        0.30 * eyes_score +
        0.25 * yawn_score +
        0.30 * phone_score +
        0.20 * head_score
    )

    # -------------------------
    # HARD SAFETY RULES
    # -------------------------

    # prolonged phone usage
    if phone_duration > 2.0:
        return 0.95, "DANGEROUS"

    # prolonged look away
    if look_away_duration > 2.0:
        return 0.90, "DANGEROUS"

    # prolonged eye closure
    if eyes_closed_duration > 2.0:
        return 0.95, "DANGEROUS"

    # NEW: prolonged yawning
    if yawn_duration > 2.5:
        return 0.75, "WARNING"

    # NEW: medium yawning
    if yawn_duration > 1.2:
        return max(risk, 0.45), "WARNING"

    # NEW: repeated yawns
    if yawn_count >= 2 and yawn_score >= 0.6:
        return max(risk, 0.55), "WARNING"

    # combined dangerous behavior
    if phone_duration > 1.0 and look_away_duration > 1.0:
        return 1.0, "DANGEROUS"

    if  eyes_closed_duration > 0.7:
        return 0.90, "DANGEROUS"

    if look_away_duration > 1.0:
        return 0.85, "DANGEROUS"

    # -------------------------
    # NORMAL THRESHOLDS
    # -------------------------
    if risk < 0.50:
        level = "SAFE"
    else:
        level = "Warning"

    return min(risk, 1.0), level
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
else:
    print("Fusion live system started. Press Q to quit.")

# time
prev_time = time.time()

# eyes temporal state
eyes_closed_duration = 0.0
last_valid_eye_state = "open"
missing_eye_frames = 0
closed_counter = 0
open_counter = 0
stable_eye_state = "open"

# yawn temporal state
yawn_prob_history = []
yawn_duration = 0.0
yawn_count = 0
currently_yawning = False
yawn_frames = 0
required_yawn_frames = 5

# phone detector temporal state
phone_history = deque(maxlen=10)
phone_duration = 0.0

# head temporal state

look_away_duration = 0.0
frame_count = 0
yawn_result = False
yawn_risk = 0.0

raw_label = None
yawn_prob = 0.0
mouth_points = []

phone_hold_time = 1.5
last_phone_time = 0
phone_result_stable = {"phone": False, "confidence": 0.0, "boxes": []}

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    frame_count += 1
    if not ret:
        print("Error reading frame.")
        break

    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time
    
    fps = 1 / dt if dt > 0 else 0

    # --------------------
    # per-frame detectors
    # --------------------
    eye_result = detect_eyes(frame)
    if frame_count % 5 == 0:
        raw_label, yawn_prob, mouth_points = detect_yawning_raw(frame)
    phone_current = detect_phone_object(frame)

    if phone_current["phone"]:
        last_phone_time = current_time
        phone_result_stable = phone_current
    else:
        if current_time - last_phone_time <= phone_hold_time:
            phone_result_stable["phone"] = True
        else:
            phone_result_stable = phone_current

    phone_object_result = phone_result_stable
    head_result = detect_head_pose(frame)

    # --------------------
    # EYES temporal logic
    # --------------------
    eye_state_raw = eye_result["eyes_label"]

    if eye_state_raw in ["open", "closed"]:
        last_valid_eye_state = eye_state_raw
        missing_eye_frames = 0
    else:
        missing_eye_frames += 1
        if missing_eye_frames <= MAX_MISSING_EYE_FRAMES:
            eye_state_raw = last_valid_eye_state
        else:
            eye_state_raw = "unknown"

    if eye_state_raw == "closed":
        closed_counter += 1
        open_counter = 0
    elif eye_state_raw == "open":
        open_counter += 1
        closed_counter = 0

    if closed_counter >= STABLE_FRAMES:
        stable_eye_state = "closed"
    elif open_counter >= STABLE_FRAMES:
        stable_eye_state = "open"

    if stable_eye_state == "closed":
        eyes_closed_duration += dt
    else:
        eyes_closed_duration = 0.0

    eyes_score = compute_eyes_score(eyes_closed_duration)

    # --------------------
    # YAWN temporal logic
    # --------------------
    if raw_label is None:
        yawn_state = "no_face"
        smoothed_prob = 0.0
        yawn_frames = 0
        yawn_duration = 0.0
        currently_yawning = False
    else:
        yawn_prob_history.append(yawn_prob)
        if len(yawn_prob_history) > 8:
            yawn_prob_history.pop(0)
        smoothed_prob = sum(yawn_prob_history) / len(yawn_prob_history)

        if smoothed_prob >= 0.40:
            yawn_frames += 1
            yawn_duration += dt
        else:
            yawn_frames = 0
            yawn_duration = 0.0
            currently_yawning = False

        if yawn_frames >= required_yawn_frames and yawn_duration >= 0.7:
            yawn_state = "yawning"
            if not currently_yawning:
                yawn_count += 1
                currently_yawning = True
        else:
            yawn_state = "not_yawning"

    yawn_score = compute_yawn_score(yawn_duration, yawn_count)

    # --------------------
    # PHONE OBJECT temporal logic
    # --------------------
    phone_history.append(phone_object_result["phone"])
    stable_phone_object = sum(phone_history) >= 3

    if stable_phone_object:
        phone_duration += dt
    else:
        phone_duration = 0.0

    phone_score = compute_phone_score(phone_duration)

    # --------------------

    # --------------------
    # HEAD temporal logic
    # --------------------
    if head_result["direction"] in ["left", "right", "down","up"]:
        look_away_duration += dt
    else:
        look_away_duration = 0.0

    head_score = compute_head_score(head_result["direction"], look_away_duration)

    # --------------------
    # FINAL RISK
    # --------------------
    final_risk, final_level = compute_final_risk(
    eyes_score,
    yawn_score,
    phone_score,
    head_score,
    phone_duration,
    look_away_duration,
    eyes_closed_duration,
    yawn_duration,
    yawn_count
)
    # --------------------
    # DISPLAY
    # --------------------
    display = eye_result.get("annotated", frame.copy())

    # phone object boxes
    for (x1, y1, x2, y2) in phone_object_result["boxes"]:
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(display, "PHONE", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # mouth points
    if mouth_points is not None and len(mouth_points) > max([LEFT_MOUTH, RIGHT_MOUTH, UPPER_INNER, LOWER_INNER]):

      for idx in [LEFT_MOUTH, RIGHT_MOUTH, UPPER_INNER, LOWER_INNER]:

        x, y = mouth_points[idx]

        cv2.circle(display, (x, y), 3, (255, 0, 0), -1)

    level_color = (0, 255, 0) if final_level == "SAFE" else (0, 255, 255) if final_level == "WARNING" else (0, 0, 255)

    lines = [
        f"FPS: {fps:.1f}"
        f"Eye State: {stable_eye_state}",
        f"Eyes Closed Dur: {eyes_closed_duration:.2f}s | Score: {eyes_score:.2f}",
        f"Yawn State: {yawn_state}",
        f"Yawn Prob: {yawn_prob:.2f} | Smooth: {smoothed_prob:.2f} | Dur: {yawn_duration:.2f}s | Score: {yawn_score:.2f}",
        f"Phone Object: {stable_phone_object} | Dur: {phone_duration:.2f}s | Score: {phone_score:.2f}",
        f"Head: {head_result['direction']} | LookAway: {look_away_duration:.2f}s | Score: {head_score:.2f}",
        f"FINAL RISK: {final_risk:.2f}",
        f"LEVEL: {final_level}"
    ]

    y_text = 30
    for i, line in enumerate(lines):
        color = level_color if i >= len(lines) - 2 else (255, 255, 0)
        if i == 0:
            color = (0, 255, 0) if stable_eye_state == "open" else (0, 0, 255)
        cv2.putText(display, line, (20, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        y_text += 32

    cv2.imshow("Fusion Risk System", display)

    key = cv2.waitKey(1) & 0xFF
    if key in [ord("q"), ord("Q"), 27]:
        break

cap.release()
cv2.destroyAllWindows()
