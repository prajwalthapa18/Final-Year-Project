import cv2
import argparse
import face_recognition
import numpy as np

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

parser = argparse.ArgumentParser()
parser.add_argument('--image')
args = parser.parse_args()

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

video = cv2.VideoCapture(args.image if args.image else 0)
padding = 20

detected_users = []  # Store known face encodings
user_ids = {}  # Map face encodings to unique user IDs
user_counter = 0  # User numbering

while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break
    
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")
    
    for faceBox in faceBoxes:
        x1, y1, x2, y2 = faceBox
        face = frame[max(0, y1 - padding):min(y2 + padding, frame.shape[0] - 1),
                     max(0, x1 - padding):min(x2 + padding, frame.shape[1] - 1)]
        
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        
        # Convert face to encoding
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_face)
        
        if encodings:
            face_encoding = encodings[0]
            matches = face_recognition.compare_faces(detected_users, face_encoding, tolerance=0.6)
            user_id = None
            
            if True in matches:
                matched_index = matches.index(True)
                user_id = user_ids[matched_index]
            else:
                user_counter += 1
                detected_users.append(face_encoding)
                user_ids[len(detected_users) - 1] = user_counter
                user_id = user_counter
                print(f'{user_id}. {gender}')
            
            cv2.putText(resultImg, f'{user_id}. {gender}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("Detecting gender", resultImg)