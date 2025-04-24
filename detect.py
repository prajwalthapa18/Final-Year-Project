#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real‑time face‑gender detection + upload to Django API
-----------------------------------------------------
Requires:
    pip install opencv-python face_recognition requests numpy
"""

import cv2
import argparse
import face_recognition
import numpy as np
import requests
import time
from pathlib import Path

# ───────────────────────────────────────── API CONFIG ──────────────────────────
API_URL = "http://127.0.0.1:8000/api/save_user/"   # Django endpoint
DETECTION_INTERVAL = 5                             # seconds between repeats

last_detection_time = {}   # user_id ► last‑sent timestamp

# ──────────────────────────────────────── HELPERS ──────────────────────────────
def save_to_backend(user_id: int, gender: str, image_bytes: bytes) -> None:
    """Send one record to Django (multipart/form‑data)."""
    now = time.time()
    if user_id in last_detection_time and now - last_detection_time[user_id] < DETECTION_INTERVAL:
        return            # too soon – skip

    files = {"image": ("face.jpg", image_bytes, "image/jpeg")}
    data  = {"user_id": user_id, "gender": gender}

    try:
        r = requests.post(API_URL, data=data, files=files, timeout=3)
        if r.status_code == 201:
            print(f"[✔] Sent  user={user_id}  gender={gender}")
            last_detection_time[user_id] = now
        else:
            print(f"[✖] Server error {r.status_code}: {r.text}")
    except requests.RequestException as e:
        print(f"[✖] Network error: {e}")

def highlight_face(net, frame, conf_threshold: float = 0.7):
    """Detect faces using OpenCV DNN; return (annotated_frame, boxes)."""
    out  = frame.copy()
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

# ───────────────────────────── MODEL FILES (Caffe & OpenCV) ────────────────────
faceProto   = "opencv_face_detector.pbtxt"
faceModel   = "opencv_face_detector_uint8.pb"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ["Male", "Female"]

# ───────────────────────────────────────── MAIN ────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Camera gender‑upload pipeline")
    parser.add_argument("--image", help="Path to image or video; omit for webcam")
    args = parser.parse_args()

    # Load networks
    faceNet   = cv2.dnn.readNet(faceModel, faceProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    # Open webcam or file
    vid = cv2.VideoCapture(args.image if args.image else 0)
    if not vid.isOpened():
        raise RuntimeError("Cannot open camera / file")

    padding = 20
    known_encodings: list[np.ndarray] = []
    id_map: dict[int, int] = {}     # index ► user_id
    next_id = 0

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

            # ───── Gender ─────
            blob = cv2.dnn.blobFromImage(face_bgr, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            gender = genderList[genderNet.forward()[0].argmax()]

            # ───── Identity (128‑d) ─────
            rgb_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb_face)
            if not encs:
                continue
            enc = encs[0]

            # Check if we've seen this person
            matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.6)
            if True in matches:
                idx = matches.index(True)
                user_id = id_map[idx]
            else:
                next_id += 1
                known_encodings.append(enc)
                id_map[len(known_encodings) - 1] = next_id
                user_id = next_id

            # ───── Upload ─────
            ok, jpeg = cv2.imencode(".jpg", face_bgr)
            if ok:
                save_to_backend(user_id, gender, jpeg.tobytes())

            # ───── Overlay ─────
            label = f"{user_id}. {gender}"
            cv2.putText(result_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Detecting gender", result_img)

    vid.release()
    cv2.destroyAllWindows()

# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
