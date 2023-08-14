
#https://github.com/techwithtim/OpenCV-Tutorials/blob/main/tutorial6.py

import numpy as np
import cv2

filename = r"C:\Users\dezos\Documents\Fibres\FibreAnalysis\Data\Raw\wool.jpg"

# img = cv2.imread('assets/chessboard.png')
img = cv2.imread(filename)

img = cv2.resize(img, (0, 0), fx=0.75, fy=0.75)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Invert the grayscale image
gray1 =  inverted_image = 255 - gray

_, gray2 = cv2.threshold(gray1, 220, 255, cv2.THRESH_TOZERO)


corners = cv2.goodFeaturesToTrack(gray2, 100, 0.5, 10)
corners = np.int0(corners)

for corner in corners:
	x, y = corner.ravel()
	cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

# for i in range(len(corners)):
# 	for j in range(i + 1, len(corners)):
# 		corner1 = tuple(corners[i][0])
# 		corner2 = tuple(corners[j][0])
# 		color = tuple(map(lambda x: int(x), np.random.randint(0, 255, size=3)))
# 		#cv2.line(img, corner1, corner2, color, 1)

cv2.imshow('Frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()