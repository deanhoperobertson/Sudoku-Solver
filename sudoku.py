#main file from which the seervice will be executed
import os
import cv2
import numpy as np
from preprocess import (
	show_empty_cells,
	clean_number_cell,
	create_grid,
	find_bounding_box,
	clean_image,
	cut_from_rect)

from extractor import identify_number, extract_sudoku


SCALE =2
PATH = os.getcwd()+"/Images/Easy.jpg"

def main():
	'''
	Diplays image sudoku puzzle image for 3 seconds.
	'''
	image = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
	image = clean_image(image)

	numbers = extract_sudoku(image)

	#image = clean_number_cell(image,grid[75],0)

	image = cv2.resize(image, (222*SCALE, 225*SCALE)) 
	cv2.imshow("Window", image)

	cv2.waitKey(3000)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()