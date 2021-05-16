#process for extraction of digits from cells
import cv2
import numpy as np

def pre_process_image(image):
	'''
	Appllies Gaussian blur, thresholding and colour inversion to an image.
	'''
	#image = cv2.GaussianBlur(image,(3,3),0)
	image = cv2.blur(image,(2,2))

	image = cv2.adaptiveThreshold(image,
		255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY,
		11,
		2)

	image = cv2.bitwise_not(image, image)

	return image


def find_corners(image):
	'''
	Find the 4 corners of the square grid.
	'''
	contours, h = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	contours = sorted(contours, key=cv2.contourArea, reverse=True)
	polygon = contours[0].tolist() #convert to list

	#find max min points to locate corners
	t1 = [i[0][0] + i[0][1] for i in polygon]
	bottom_right,top_left = t1.index(max(t1)), t1.index(min(t1))

	t2 = [i[0][0] - i[0][1] for i in polygon]
	top_right, bottom_left = t2.index(max(t2)), t2.index(min(t2))

	return [polygon[top_left], polygon[top_right], polygon[bottom_left], polygon[bottom_right]]


