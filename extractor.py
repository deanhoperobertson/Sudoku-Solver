#process for extraction of digits from cells
import cv2

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

