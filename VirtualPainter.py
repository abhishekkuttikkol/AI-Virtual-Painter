import cv2
import numpy
import HandTrackingModule as htm
import os
import time

################################
drawThickness = 15
eraserThickness = 50
###############################
folderPath = "AI Virtual Painter\header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]
detector = htm.handDetector(detectionCon=0.85)
lmList = []
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
# cap.open('http://192.168.1.199:8080/video') 
imageCanvas = numpy.zeros((720, 1280, 3), numpy.uint8)

while cap.isOpened():
    success, img = cap.read()
    img = detector.findhands(img)
    # img = cv2.addWeighted(img, 0.5, imageCanvas, 0.5, 0)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)
    
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
    
        fingers = detector.fingersUp()
        
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection Mode")

            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)

                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                
                elif 1050 < x1 < 1200:
                    drawColor = (0, 0, 0)
                    header = overlayList[3]

            cv2.rectangle(img, (x1, y1-35), (x2, y2 + 25), drawColor, cv2.FILLED)
                
        
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imageCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)

            cv2.line(img, (xp, yp), (x1, y1), drawColor, drawThickness)
            cv2.line(imageCanvas, (xp, yp), (x1, y1), drawColor, drawThickness)
            print("Drawing mode")
            xp, yp = x1, y1
        
    imageGray = cv2.cvtColor(imageCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imageGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imageCanvas)

    img[0:125, 0:1280] = header
    cv2.imshow("Virtual Painter", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
