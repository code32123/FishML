import cv2
key = 255

cap = cv2.VideoCapture('http://192.168.1.56:8080/video')
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1);
success, img = cap.read()
cv2.imshow("img", img)

while key == 255:
	key = cv2.waitKey(1) & 0xFF
print(key)