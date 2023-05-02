import numpy as np
import cv2
import cv2 as cv
import mediapipe as mp
import matplotlib.pylab as plt
import pytesseract
import time
from playsound import playsound

FileNameCounter = int(input("Start from: "))

LOCATIONS = { ## X, Y, W, H
	"Energy": (10, 6, 96, 25)
}

pytesseract.pytesseract.tesseract_cmd = './tesseract.exe'
tessdata_dir_config = r'--tessdata-dir "./tesseract-5.0.0-rc1/tessdata"'

cap = cv2.VideoCapture('http://192.168.1.56:8080/video')
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1);

def crop_minAreaRect(img, rect):
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	width = int(rect[1][0])
	height = int(rect[1][1])
	src_pts = box.astype("float32")
	dst_pts = np.array([[0, height-1],
						[0, 0],
						[width-1, 0],
						[width-1, height-1]], dtype="float32")
	M = cv2.getPerspectiveTransform(src_pts, dst_pts)
	warped = cv2.warpPerspective(img, M, (width, height))
	if width > height:
		warped = cv2.rotate(warped, cv2.cv2.ROTATE_90_CLOCKWISE)
	return warped

def OCRinRect(img, x, y, w, h):
	rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
	cropped = img[y:y + h, x:x + w]
	text = pytesseract.image_to_string(img_croped)
	return text

while True:
	success, img = cap.read()
	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 127, 255, 0)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# Largest contour
	areas = [cv2.contourArea(c) for c in contours]
	max_index = np.argmax(areas)
	cnt=contours[max_index]

	rect = cv2.minAreaRect(cnt)
	img_croped = crop_minAreaRect(img, rect)


	key = cv2.waitKey(1) & 0xFF

	# KEYBOARD INTERACTIONS
	# Q is quit

	if key == ord('q'):
		cv2.destroyAllWindows()
		break
	elif key == 32:
		cv2.imwrite(f"./Images/Unfiled/{FileNameCounter}.png", img_croped)
		print("Wrote file:", FileNameCounter)
		FileNameCounter += 1
		playsound('C:\\Users\\jimmy\\Desktop\\Coding\\Python\\ML\\Ding.mp3')
		pass
	elif key != 255:
		print("No Bindings For", key, chr(key))

	cv2.drawContours(img, [cnt], 0, (255,0,0), 3)
	cv2.imshow("LIVE", img)
	cv2.imshow("CROP", img_croped)
