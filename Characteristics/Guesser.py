import numpy as np
import cv2
import cv2 as cv
# import mediapipe as mp
# import matplotlib.pylab as plt
# import pytesseract
# import time
# from playsound import playsound
import os
import time





from PIL import ImageGrab
import pyautogui
import keyboard

from colorama import Fore, Style, init
init()
import ColoredFont as cf

import GuesserTF
import TFTrainer

MAXIMUM_FILES = 500

BWTHRESH = 80
ImageX, ImageY = 826, 365
button1 = (1727, 1571) # Not Fish
button2 = (2052, 1595) # Fish

screenScale = 2

TextPoint = (50,650)

keyLoopDelay = 12

# img = ImageGrab.grab(bbox=(0, 1000, 100, 1100))
# cap = cv2.VideoCapture('http://192.168.1.66:8080/video')
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1);

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
	# if width > height:
	warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
	warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
	return warped

class TooManyFiles(Exception):
	print("There are too many files - names exceed", cf.mkRed(str(MAXIMUM_FILES)))
	print("To resolves, adjust the program variable MAXIMUM_FILES")
	pass

# t: train
# r: reset then train
# y: train {input} epochs
# q: quit
# m: move box
# s: save as {input}
# a: save as {guessed}
# x: execute
# w: switch screen scale between 1 and 2
# b: set position of button 1
# n: set position of button 2

# g: guess
# space: guess and click
# 0: preferable alternate to space
# 9: guess and click, locks down then releases

def cropToSquare(GrayImage, ColImage):
	ret, thresh = cv2.threshold(GrayImage, 127, 255, 0)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# Largest contour
	areas = [cv2.contourArea(c) for c in contours]
	if len(areas) > 0:
		max_index = np.argmax(areas)
		cnt=contours[max_index]

		rect = cv2.minAreaRect(cnt)
		img_cropped = crop_minAreaRect(ColImage, rect)
		img_cropped = cv2.resize(img_cropped,dsize=(200,200), interpolation = cv2.INTER_CUBIC)
	else:
		cnt = []
		img_cropped = ColImage
	return img_cropped, cnt

def saveImage(img, folder):
	files = os.listdir(f"./Images/{folder}/")
	file = 0
	while (str(file) + ".png") in files:
		if file > MAXIMUM_FILES:
			raise TooManyFiles
		file += 1
	cv2.imwrite(f"./Images/{folder}/{file}.png", img)
	print("Wrote", f"./Images/{folder}/{file}.png")

def inputAction(actionName, strToPrint):
	global comingInput, typing, PreviousAction, KeypressMode
	comingInput = ""
	typing = ""
	PreviousAction = actionName
	KeypressMode = False
	print(strToPrint, end="", flush=True)

def onInputFinish(actionName):
	global comingInput, typing, PreviousAction
	if PreviousAction == actionName and typing != 0:
		PreviousAction = 0
		comingInput = typing
		typing = 0
		return True
	return False

def checkKeyboard(keyToCheck, localKey='undefined', mode='undefined', doOrd=True):
	global i
	global local, key
	if (local and mode == 'undefined') or mode == 'local':
		if localKey != 'undefined':
			keyToCheck = localKey
		if doOrd:
			return (key == ord(keyToCheck))
		else:
			return (key == keyToCheck)
	else:
		if not keyToCheck in keyPressChecker:
			keyPressChecker[keyToCheck] = (False, 0)
		if keyboard.is_pressed(keyToCheck) and not keyPressChecker[keyToCheck][0]:
			keyPressChecker[keyToCheck] = (True, i)
			# print(keyToCheck, localKey, "FOUND KEY")
			return True
		elif (not keyboard.is_pressed(keyToCheck)) or keyPressChecker[keyToCheck][1]+keyLoopDelay <= i:
			keyPressChecker[keyToCheck] = (False, 0)
		return False

def guess(outText = True, useGlobal = True):
	global img_cropped
	if useGlobal:
		global catagory, probability, probabilityStr

	outImg = cv.cvtColor(img_cropped, cv.COLOR_BGR2RGB)
	np_image_data = np.asarray(outImg)
	np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
	# np_final = np.expand_dims(np_image_data,axis=0)
	catagory, probability = GuesserTF.guess(outImg)
	if probability >= 0.9:
		probabilityStr = cf.mkGreen(str(probability))
	elif probability >= 0.5:
		probabilityStr = cf.mkYellow(str(probability))
	else:
		probabilityStr = cf.mkRed(str(probability))

	if outText:
		print(catagory, probabilityStr)
	if not useGlobal:
		return catagory, probability

def clickButton(catagoryToClick = 'undefined', needProb = True, doSave = False):
	global updateFrame, button1, button2, local, restriction
	if catagoryToClick == 'undefined':
		global catagory, probability
		guess()
	else:
		probability = 1
		catagory = catagoryToClick
	if probability > 0.98 or not needProb:
		if doSave:
			saveImage(img_cropped, catagory)
		if catagory[int(restriction[0]=="Shape")] == restriction[1][0]:
			pyautogui.click(x=button2[0], y=button2[1])
			updateFrame = 1
		else:
			pyautogui.click(x=button1[0], y=button1[1])
			updateFrame = 1
	else:
		print("Confidence less than 98% - no action taken")
		return False
	return True

# Options BC BR BT, GC GR GT, RC RR RT, PC PR PT, YC YR YT
i = input("Feature: ")
restriction = ("Shape" if i in ["Circular", "Rectangle", "Triangle"] else "Color", i)
i=0
catagory = ""
probability = 0
updateFrame = 1
keyPressChecker = {}
local = True
KeypressMode = True
typing = ""
PreviousAction = 0
lockDown = False

def scale(tuple, scalar):
	return (tuple[0]*scalar, tuple[1]*scalar)


while True:
	if not KeypressMode:
		key = cv2.waitKey(1) & 0xFF
		if key == 13: # Enter
			print(chr(key))
			KeypressMode = True
		elif key == 8: # Backspace
			print(chr(key) + " " + chr(key), end="", flush=True)
			typing = typing[:-1]
		elif key != 255:
			print(chr(key), end="", flush=True)
			typing += chr(key)
		continue
		
	if lockDown:
		time.sleep(0.5)
	# success, img = cap.read()
	img = np.array(ImageGrab.grab(bbox=((ImageX*screenScale), (ImageY*screenScale), (ImageX+300)*screenScale, (ImageY+350)*screenScale)))
	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgGray, 127, 255, 0)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# (thresh, imgBW) = cv2.threshold(imgGray, BWTHRESH, 255, cv2.THRESH_BINARY)


	useImg = imgRGB # Could use imgBW, imgGrey, or img

	img_cropped = ~cropToSquare(~cropToSquare(imgGray, imgGray)[0], ~cropToSquare(imgGray, useImg)[0])[0]
		
	if lockDown:
		clickButton()
			
		printColor = (0,255,0) if probability > 0.98 else (0,200,255) if probability > 0.90 else (0,0,255)
		cv2.putText(useImg, text=catagory, org=TextPoint, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=printColor, thickness=3)
		cv2.imshow("Image", useImg)
		
		key = cv2.waitKey(1) & 0xFF
		if checkKeyboard('9'):
			lockDown = not lockDown
			print("AutoLock state change to",lockDown)
		continue

	key = cv2.waitKey(1) & 0xFF
	if checkKeyboard('q', mode='local'):
		cv2.destroyAllWindows()
		break

	if onInputFinish("SaveAs"):
		saveImage(img_cropped, comingInput)
		clickButton(comingInput)
	elif onInputFinish("Execute"):
		try:
			exec(comingInput)
		except:
			print(cf.mkRed("FAIL"))
	elif onInputFinish("TrainAs"):
		TFTrainer.train(True, int(comingInput))
	elif onInputFinish("ReTrainAs"):
		TFTrainer.train(False, int(comingInput))

	if checkKeyboard('l', mode='local'):
		local = not local
		print("Key context mode changed: LOCAL=" + str(local))

	elif checkKeyboard('a'):
		clickButton(needProb=False, doSave = True)
	elif checkKeyboard('s'):
		inputAction("SaveAs", "Save to: ")

	elif checkKeyboard('k'):
		updateFrame = 1


	elif checkKeyboard('x'):
		inputAction("Execute", "Execute: ")


	elif checkKeyboard('d'):
		pyautogui.click(x=button1[0], y=button1[1])
		updateFrame = 1
	elif checkKeyboard('f'):
		pyautogui.click(x=button2[0], y=button2[1])
		updateFrame = 1


	elif checkKeyboard('w'):
		screenScale = 2 if screenScale == 1 else 1
	elif checkKeyboard('m'):
		ImageX,ImageY = scale(pyautogui.position(), 1/screenScale)
		print("Img", ImageX,ImageY, "To adjust this permanentaly, edit the screenScale variable")
	elif checkKeyboard('b'):
		button1 = pyautogui.position()
		print("B1", button1, "To adjust this permanentaly, edit the button1 variable")
	elif checkKeyboard('n'):
		button2 = pyautogui.position()
		print("B2", button2, "To adjust this permanentaly, edit the button2 variable")


	elif checkKeyboard('t', mode='local'):
		TFTrainer.train(True, 5)
	elif checkKeyboard('r', mode='local'):
		TFTrainer.train(False, 5)
	elif checkKeyboard('R', mode='local'):
		inputAction("ReTrainAs", "Re-Train: ")
	elif checkKeyboard('T', mode='local'):
		inputAction("TrainAs", "Train: ")


	elif checkKeyboard('9'):
		lockDown = True
		local = False
		print("AutoLock state change to",lockDown)
	# elif lockDown:
	# 	if i%keyLoopDelay == 0:
	# 		lockDown = clickButton()
	# 		if not lockDown:
	# 			print('\a')
	elif checkKeyboard('g'):
		guess()

	elif checkKeyboard('space', ' ') or checkKeyboard("0"):
		clickButton()
	elif checkKeyboard('.'):
		clickButton(doSave = True, needProb = False)

	if updateFrame != False:
		if updateFrame >= 4:
			guess(outText=False)
			updateFrame = False
		updateFrame +=1

	# useImg = cv2.resize(useImg,dsize=(400,400))
	printColor = (0,255,0) if probability > 0.98 else (0,200,255) if probability > 0.90 else (0,0,255)
	cv2.putText(useImg, text=catagory, org=TextPoint, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=printColor, thickness=3)
	cv2.imshow("Image", useImg)
	# cv2.imshow("LIVE1", img_cropped1)
	# cv2.imshow("LIVE", img_cropped)
	i+=1
