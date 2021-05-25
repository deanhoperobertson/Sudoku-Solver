#main file from which the seervice will be executed
import os
import cv2
import numpy as np
from preprocess import (
	show_empty_cells,
	clean_number_cell,
	create_grid,
	find_bounding_box,
	clean_image)

from extractor import identify_number #extract_sudoku

SCALE =2
PATH = os.getcwd()+"/Images/Easy.jpg"

def main():
	'''
	Diplays image sudoku puzzle image for 3 seconds.
	'''
	image = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
	image = clean_image(image)

	#image = show_empty_cells(image)

	# numbers = extract_sudoku(image)
	# print(numbers)

	grid = create_grid(image)

	image = clean_number_cell(image,grid[11])

	number = identify_number(image)
	print(number)

	image = cv2.resize(image, (222*SCALE, 225*SCALE)) 
	cv2.imshow("Window", image)

	cv2.waitKey(3000)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()