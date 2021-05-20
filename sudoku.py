#main file from which the seervice will be executed
import os
import cv2
import numpy as np
from preprocess import (
	pre_process_image,
	find_corners,
	wrap_crop_image,
	create_grid,
	extract_digit,
	insert_circle)

SCALE =2
PATH = os.getcwd()+"/Images/Easy.jpg"

def main():
	'''
	Diplays image sudoku puzzle image for 3 seconds.
	'''
	image = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
	image = pre_process_image(image)
	corners = find_corners(image)
	image = wrap_crop_image(image,corners)

	grid = create_grid(image)

	image=extract_digit(image,grid[0])

	# image = insert_circle(image, grid[0])

	image = cv2.resize(image, (222*SCALE, 225*SCALE)) 
	cv2.imshow("Window", image)

	cv2.waitKey(2000)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()