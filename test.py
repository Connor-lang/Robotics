from imutils.video import VideoStream
import imutils
import time 
import cv2
import os
import argparse
import numpy as np
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
# from pygame import mixer 

# mixer.init()
# sound = mixer.Sound('alarm.WAV')

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
    default="face_detector",
    help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
    default="mask_detector.model",
    help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
#set src to 2 on robot
# vs = VideoStream(src=2).start() 
time.sleep(1.0)

while True:
    frame = vs.read()

    image_height, image_width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300),
                                mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)
    faceNet.setInput(blob)
    results = faceNet.forward()

    locs = []
    faces = []
    preds = []

    for face in results[0][0]:
        face_confidence = face[2]
        if face_confidence > args["confidence"]:
            bbox = face[3:]

            x1 = max(0, int(bbox[0] * image_width))
            y1 = max(0, int(bbox[1] * image_height))
            x2 = min(image_width - 1 ,int(bbox[2] * image_width))
            y2 = min(image_height- 1, int(bbox[3] * image_height))
            locs.append((x1, y1, x2, y2))

            face = frame[y1:y2, x1:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            faces.append(face)
    
    if len(faces) > 0:
        for face in faces:  
            preds.append(maskNet.predict(face))
    
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred[0]
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        if mask > withoutMask:
            cv2.putText(frame, 'FACE-MASK ON', (150,40), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0,0,255), 3)
        else:
            if mask < withoutMask:
                # sound.play()
                cv2.putText(frame, 'FACE-MASK NOT ON', (150,40), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0,0,255), 3)
            else:
                cv2.putText(frame, 'FACE-MASK ON', (350,40), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0,0,255), 3)
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
