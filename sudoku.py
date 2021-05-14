#main file from which the seervice will be executed
import os
import cv2
from extractor import pre_process_image

SCALE =2
PATH = os.getcwd()+"/Images/Easy.jpg"

def show_image():
	'''
	Diplays image sudoku puzzle image for 3 seconds.
	'''

	image = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
	image = pre_process_image(image)
	image = cv2.resize(image, (222*SCALE, 225*SCALE)) 
	cv2.imshow("Window", image)

	cv2.waitKey(2000)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	show_image()