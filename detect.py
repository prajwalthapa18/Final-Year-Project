#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced real-time face-gender detection + upload to Django API
-------------------------------------------------------------
Fixed bug with last_detection_time not being defined
"""

import cv2
import argparse
import face_recognition
import numpy as np
import requests
import time
from pathlib import Path
from collections import defaultdict

# ───────────────────────────────────────── API CONFIG ──────────────────────────
API_URL = "http://127.0.0.1:8000/api/save_user/"   # Django endpoint
DETECTION_INTERVAL = 5                             # seconds between repeats
MIN_CONFIDENCE = 0.6                              # Minimum face recognition confidence
GENDER_CONFIDENCE_THRESHOLD = 0.8                  # Minimum gender confidence to accept

# ─────────────────────────────────────── TRACKING STATE ────────────────────────
face_tracker = {
    'known_encodings': [],         # List of known face encodings
    'id_map': {},                  # index → user_id
    'last_seen': defaultdict(float),  # user_id → last seen timestamp
    'gender_buffer': defaultdict(list),  # user_id → list of recent gender detections
    'next_id': 1,                  # Next available user ID
    'uploaded_ids': set(),         # Track which IDs we've already uploaded
    'last_detection_time': defaultdict(float)  # Track last upload time per user
}

# ──────────────────────────────────────── HELPERS ──────────────────────────────
def save_to_backend(user_id: int, gender: str, image_bytes: bytes) -> None:
    """Send one record to Django (multipart/form-data) if conditions are met."""
    now = time.time()
    
    # Skip if we've already uploaded this user
    if user_id in face_tracker['uploaded_ids']:
        print(f"[→] Detected user={user_id} gender={gender} (already uploaded)")
        return
    
    # Skip if we've sent this recently
    if now - face_tracker['last_detection_time'].get(user_id, 0) < DETECTION_INTERVAL:
        print(f"[→] Detected user={user_id} gender={gender} (waiting {DETECTION_INTERVAL}s)")
        return
    
    # Only send if we have consistent gender detection
    gender_history = face_tracker['gender_buffer'][user_id]
    if len(gender_history) < 3:  # Need at least 3 samples
        print(f"[→] Detected user={user_id} gender={gender} (collecting samples: {len(gender_history)}/3)")
        return
        
    # Check if we have consistent gender detection
    from collections import Counter
    gender_counts = Counter(gender_history)
    most_common = gender_counts.most_common(1)[0]
    
    # Only proceed if we have a clear majority (>= 66%) and it matches current detection
    if most_common[1] < len(gender_history) * 0.66 or most_common[0] != gender:
        print(f"[→] Detected user={user_id} gender={gender} (inconsistent samples)")
        return

    files = {"image": ("face.jpg", image_bytes, "image/jpeg")}
    data = {"user_id": user_id, "gender": gender}

    try:
        r = requests.post(API_URL, data=data, files=files, timeout=3)
        if r.status_code == 201:
            print(f"[✔] UPLOADED user={user_id} gender={gender} (confidence: {most_common[1]/len(gender_history):.0%})")
            face_tracker['last_detection_time'][user_id] = now
            face_tracker['uploaded_ids'].add(user_id)
            face_tracker['gender_buffer'][user_id] = []  # Clear buffer after successful send
        else:
            print(f"[✖] Server error {r.status_code}: {r.text}")
    except requests.RequestException as e:
        print(f"[✖] Network error: {e}")

def highlight_face(net, frame, conf_threshold: float = 0.7):
    """Detect faces using OpenCV DNN; return (annotated_frame, boxes)."""
    out = frame.copy()
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(out, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            boxes.append([x1, y1, x2, y2])
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), max(1, h // 150))
    return out, boxes

def get_face_identity(face_encoding):
    """Match face against known faces or assign new ID."""
    if not face_tracker['known_encodings']:
        # First face - assign new ID
        new_id = face_tracker['next_id']
        face_tracker['known_encodings'].append(face_encoding)
        face_tracker['id_map'][0] = new_id
        face_tracker['next_id'] += 1
        return new_id
    
    # Compare with known faces
    distances = face_recognition.face_distance(face_tracker['known_encodings'], face_encoding)
    best_match_idx = np.argmin(distances)
    best_distance = distances[best_match_idx]
    
    if best_distance < (1 - MIN_CONFIDENCE):
        # Known face
        user_id = face_tracker['id_map'][best_match_idx]
    else:
        # New face
        new_id = face_tracker['next_id']
        face_tracker['known_encodings'].append(face_encoding)
        face_tracker['id_map'][len(face_tracker['known_encodings']) - 1] = new_id
        face_tracker['next_id'] += 1
        user_id = new_id
    
    return user_id

# ───────────────────────────── MODEL FILES (Caffe & OpenCV) ────────────────────
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ["Male", "Female"]

# ───────────────────────────────────────── MAIN ────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Camera gender-upload pipeline")
    parser.add_argument("--image", help="Path to image or video; omit for webcam")
    args = parser.parse_args()

    # Load networks
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    # Open webcam or file
    vid = cv2.VideoCapture(args.image if args.image else 0)
    if not vid.isOpened():
        raise RuntimeError("Cannot open camera / file")

    padding = 20

    while cv2.waitKey(1) < 0:
        has_frame, frame = vid.read()
        if not has_frame:
            break

        result_img, face_boxes = highlight_face(faceNet, frame)
        if not face_boxes:
            cv2.imshow("Detecting gender", result_img)
            continue

        for box in face_boxes:
            x1, y1, x2, y2 = box
            face_bgr = frame[max(0, y1 - padding):min(y2 + padding, frame.shape[0]-1),
                             max(0, x1 - padding):min(x2 + padding, frame.shape[1]-1)]

            # ───── Gender Detection ─────
            blob = cv2.dnn.blobFromImage(face_bgr, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            gender_preds = genderNet.forward()
            gender_idx = gender_preds[0].argmax()
            gender_confidence = gender_preds[0][gender_idx]
            gender = genderList[gender_idx]
            
            # Skip if gender confidence is low
            if gender_confidence < GENDER_CONFIDENCE_THRESHOLD:
                continue

            # ───── Face Identity ─────
            rgb_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb_face)
            if not encs:
                continue
            enc = encs[0]
            
            user_id = get_face_identity(enc)
            now = time.time()
            
            # Update tracking information
            face_tracker['last_seen'][user_id] = now
            face_tracker['gender_buffer'][user_id].append(gender)
            
            # Keep only recent gender detections (last 5 seconds)
            face_tracker['gender_buffer'][user_id] = [
                g for g in face_tracker['gender_buffer'][user_id] 
                if now - face_tracker['last_seen'][user_id] < 5
            ]
            
            # ───── Upload Decision ─────
            ok, jpeg = cv2.imencode(".jpg", face_bgr)
            if ok:
                save_to_backend(user_id, gender, jpeg.tobytes())

            # ───── Display Overlay ─────
            status = "UPLOADED" if user_id in face_tracker['uploaded_ids'] else "DETECTED"
            label = f"ID:{user_id} {gender} ({gender_confidence:.0%}) [{status}]"
            color = (0, 255, 0) if status == "UPLOADED" else (0, 165, 255)  # Green for uploaded, orange for detected
            cv2.putText(result_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Detecting gender", result_img)

    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()