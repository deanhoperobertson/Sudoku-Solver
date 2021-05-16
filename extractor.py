#process for extraction of digits from cells
import cv2
import numpy as np
import math
from typing import Any, List


def pre_process_image(image: np.ndarray):
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


def find_corners(image: np.ndarray):
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


def get_distance(pt1: Any, pt2: Any):
	'''
	Calculates the distance between 2 points.
	'''
	side_1 = pt2[0][0] - pt1[0][0]
	side_2 = pt2[0][1] - pt1[0][1]
	return math.sqrt(side_1**2 + side_2**2)


def wrap_image(image: np.ndarray, corners: List):
	'''
	Wrap the image to create a birds eye of the puzzle and cut out any space around the grid.
	'''
	top_left, top_right, bottom_left, bottom_right = corners[0], corners[1], corners[2], corners[3]

	src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

	max_side = max([
		get_distance(top_left, top_right),
		get_distance(top_right, bottom_right),
		get_distance(bottom_right, bottom_left),
		get_distance(bottom_left,top_left)
		])

	#crate a new rectangle which starts in the top left pixel with equal length of max sides
	rect = np.array([[0, 0], [max_side - 1, 0], [max_side - 1, max_side - 1], [0, max_side - 1]], dtype='float32')

	m = cv2.getPerspectiveTransform(src,rect)

	return cv2.warpPerspective(image, m, (int(max_side), int(max_side)))

