import cv2
import numpy as np
import time
import Hand_Tracking_Module as htm


cap = cv2.VideoCapture(0) # 1 ?

pTime = 0
color = (0,0,255)
xp, yp = 0, 0
thickness = 10
eraser_thickness = 30

canvas = np.zeros((480, 640, 3), np.uint8)

detector = htm.HandDetector()

while 1:
	success, img = cap.read()
	img = cv2.flip(img, 1)

	img = detector.find_hands(img, draw=False)
	lm_list = detector.find_position(img, draw=False)

	if lm_list:
		x1, y1 = lm_list[8][1:]
		x2, y2 = lm_list[12][1:]
		fingers = detector.fingers_up()
		if fingers[0] and fingers[1]:
			xp, yp = 0, 0
			if y1 < 110:
				if 10 <= x1 <= 110:
					color = (0,0,255)
				elif 120 <= x1 <= 220:
					color = (0,255,0)
				elif 230 <= x1 <= 330:
					color = (255,0,0)
				elif 340 <= x1 <= 440:
					color = (0,0,0)
			cv2.circle(img,(x1,y1), 10, color, cv2.FILLED)
		elif fingers[0] and not fingers[1]:
			cv2.circle(img,(x1,y1), 15, color, cv2.FILLED)
			if not xp and not yp:
				xp, yp = x1, y1
			if color == (0,0,0):
				cv2.line(canvas, (xp,yp), (x1,y1), color, eraser_thickness)
			else:
				cv2.line(canvas, (xp,yp), (x1,y1), color, thickness)
			xp, yp = x1, y1

	cv2.rectangle(img,(10,10), (110,110), (0,0,255), 2)
	cv2.rectangle(img,(120,10), (220,110), (0,255,0), 2)
	cv2.rectangle(img,(230,10), (330,110), (255,0,0), 2)
	cv2.rectangle(img,(340,10), (440,110), (255,255,255), 2)

	cTime = time.time()
	fps = 1/(cTime - pTime)
	pTime = cTime

	grey = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
	_, inverse = cv2.threshold(grey, 50, 255, cv2.THRESH_BINARY_INV)
	inverse = cv2.cvtColor(inverse, cv2.COLOR_GRAY2BGR)
	img = cv2.bitwise_and(img, inverse)
	img = cv2.bitwise_or(img, canvas)
		#img = cv2.addWeighted(img,0.5,canvas,0.5,0)

	cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
	cv2.imshow("Image", img)
	cv2.waitKey(1)

