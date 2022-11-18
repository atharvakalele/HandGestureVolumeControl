import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils



while True:
    success, img = cap.read()
    image = cv2.flip(img,1)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    # print(results.multi_hand_landmarks)
    if  results.multi_hand_landmarks:
       for handLMS in results.multi_hand_landmarks:
           for id,lm in enumerate(handLMS.landmark):
               h, w, c = image.shape
               cx,cy = int(lm.x * w),int(lm.y * h)
               print(id,cx,cy)
               if(id == 0):
                   cv2.circle(image,(cx,cy),25,(255,0,255),cv2.FILLED)
           mpDraw.draw_landmarks(image, handLMS, mpHands.HAND_CONNECTIONS)
    cv2.imshow("Image",image)
    cv2.waitKey(1)

