from imutils.object_detection import non_max_suppression
from imutils import paths
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import cv2
import time
# from pygame import mixer 

# mixer.init()
# sound = mixer.Sound('alarm.WAV')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
vs = VideoStream(src=0).start()
#set src to 2 on robot
# vs = VideoStream(src=2).start() 
time.sleep(1.0)
threshold = int(input("Enter your Threshold value: "))
print("\n")
print('[INFO] Opening Web Cam.')
while True:
    frame = vs.read()
    rects, weights = hog.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.05)
    rects = np.array([[x, y, x+w, y+h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    person = 1
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 0, 255), 2)
        cv2.putText(frame, f'person {person}', (xA,yA), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
        person += 1
    if (person -1) > threshold:
        #sound.play()
        cv2.putText(frame, 'CROWD DETECTED', (350,40), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0,0,255), 3)
    else:
        cv2.putText(frame, 'CROWD NOT DETECTED', (350,40), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0,0,255), 3)

    cv2.putText(frame, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.putText(frame, f'Total Persons : {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.imshow('Output', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()