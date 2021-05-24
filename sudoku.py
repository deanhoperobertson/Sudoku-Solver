#main file from which the seervice will be executed
import os
import cv2
import numpy as np
from preprocess import (
	pre_process_image,
	wrap_crop_image,
	show_empty_cells,
	clean_number_cell,
	create_grid)

SCALE =2
PATH = os.getcwd()+"/Images/Easy.jpg"

def main():
	'''
	Diplays image sudoku puzzle image for 3 seconds.
	'''
	image = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
	image = pre_process_image(image)
	image = wrap_crop_image(image)

	grid = create_grid(image)

	image = clean_number_cell(image,grid[1])

	# image = show_empty_cells(image)
	# cv2.imwrite('empty_cells.jpg',image)
	image = cv2.resize(image, (222*SCALE, 225*SCALE)) 
	cv2.imshow("Window", image)

	cv2.waitKey(3000)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()