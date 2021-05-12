import os
import cv2
from PIL import Image

PATH = os.getcwd()+"/Images/Easy.jpg"

def show_image():
	#Image.open(PATH).show()

	image = cv2.imread(PATH)
	cv2.imshow("Window", image)
	cv2.waitKey(2000)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	show_image()