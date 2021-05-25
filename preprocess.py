#process for extraction of digits from cells
import cv2
import numpy as np
import math
from typing import Any, List, Tuple


def pre_process_image(image: np.ndarray) -> np.ndarray:
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
        15)

    image = cv2.bitwise_not(image, image)
    return image


def find_corners(image: np.ndarray) -> np.ndarray:
    '''
    Find the 4 corners of the biggest square in the image.
    '''
    contours, h = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0].tolist() #convert to list

    #find max min points to locate corners
    t1 = [i[0][0] + i[0][1] for i in polygon]
    bottom_right,top_left = t1.index(max(t1)), t1.index(min(t1))

    t2 = [i[0][0] - i[0][1] for i in polygon]
    top_right, bottom_left = t2.index(max(t2)), t2.index(min(t2))

    return polygon[top_left], polygon[top_right], polygon[bottom_left], polygon[bottom_right]


def get_distance(pt1: List[int], pt2: List[int]) -> float:
    '''
    Calculates the distance between 2 coordinates.
    '''
    side_1 = pt2[0][0] - pt1[0][0]
    side_2 = pt2[0][1] - pt1[0][1]
    return math.sqrt(side_1**2 + side_2**2)


def wrap_crop_image(image: np.ndarray) -> np.ndarray:
    '''
    Wrap and crop the image to create a birds eye of the puzzle and cut out any space around the grid.
    '''
    top_left, top_right, bottom_left, bottom_right = find_corners(image)

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


def create_grid(image: np.ndarray) -> List[int]:
    '''
    Find the coordinates of all 81 cells that make up the grid.
    
    Method:
    pt1------|
    |        |
    |        |
    |-------pt2

    '''
    one_side = image.shape[0]
    cell = one_side/9
    output = []

    for x in range(9):
        for y in range(9):
            pt1 = [(x*cell),(y*cell)]
            pt2 = [((x+1)*cell),((y+1)*cell)]
            output.append([pt1,pt2])
    return output


def cut_from_rect(img: np.ndarray, rect: List[int]) -> np.ndarray:
    '''
    Cuts the image using the top left and bottom right points.
    '''
    return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]


def has_number(image_cell: np.ndarray) -> bool:
    '''
    Trim borders of image and detect presence of number.
    '''
    one_side=image_cell.shape[0]
    trim = 2
    image_cell = cut_from_rect(image_cell,[[trim, trim], [one_side-trim, one_side-trim]])

    #count number of black pixels
    n_black_pix = np.sum(image_cell == 0)

    if int(n_black_pix) > 290:
        return False
    else:
        return True


def find_center(square: List[int]) -> Tuple[int]:
    '''
    Find the global coordinates of the centre of the cell.
    '''
    pt1 = square[0]
    pt2 = square[1]

    x = round(pt1[0]+((pt2[0]-pt1[0])/2))
    y = round(pt1[1]+((pt2[1]-pt1[1])/2))

    return (x,y)


def show_empty_cells(image: np.ndarray) -> np.ndarray:
    ''''
    Populaates the empty grid cells with a red dot.
    '''
    squares = create_grid(image)

    no_numbers = []

    for square in squares:
        image_cell = cut_from_rect(image, square)

        if has_number(image_cell) == False:
            no_numbers.append(square)

    #convert image back to colour
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

    for cell in no_numbers:
        center = find_center(cell)
        image = cv2.circle(image, center, radius=5, color=(0,0,255), thickness=-1)
    return image


def find_bounding_box(image: np.ndarray) -> np.ndarray:
    '''
    Find the bounding box around the digit in cell.
    '''
    height, width = image.shape[:2]
    top, bottom, left, right = height, 0, width, 0

    for x in range(width):
        for y in range(height):
            if image.item(y, x) >=250:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right

    pad = 3
    box = [[left-pad, top-pad], [right+pad, bottom+pad]]
    image = cut_from_rect(image,box)

    return image


def clean_number_cell(image: np.ndarray, grid) -> np.ndarray:
    '''
    Prepares the image cell with numbers for digit recogntion.
    - remove background
    - lift digit off backdrop
    - invert grey scale
    - background all white pixels
    '''

    image = cut_from_rect(image,grid)

    height, width = image.shape[:2]

    max_area = 0
    seed_point = (None, None)
    trim = 3

    #loop through zoomed in centre of image to finde the main feature
    #colour the main feature with grey
    for x in range(0+trim, width-trim):
        for y in range(0+trim, height-trim):
            # Only operate on light or white squares
            if image.item(y, x) >= 130 :
                area = cv2.floodFill(image, None, (x, y), 64)
                if area[0] > max_area:  # Gets the maximum bound area which should be the grid
                    max_area = area[0]
                    seed_point = (x, y)


    #Anyting lighter then grey is filled with black pixels
    for x in range(0, width):
        for y in range(0, height):
            if image.item(y, x) >= 100:
                cv2.floodFill(image, None, (x, y), 0)

    #create mask
    mask = np.zeros((image.shape[0]+2, image.shape[1]+2), np.uint8)


    # Highlight the main feature and fill with white pixels
    if all([p is not None for p in seed_point]):
        cv2.floodFill(image, mask, seed_point, 255)

    #black out every pixel darker than white (ie. not the main feature)
    for x in range(0, width):
        for y in range(0, height):
            if image.item(y, x) <= 250: 
                cv2.floodFill(image, mask, (x, y), 0)

    #find boudning box and pad scale image
    image = find_bounding_box(image)
    return  image


def clean_image(image: np.ndarray) -> np.ndarray:
    '''
    '''
    #preprocess
    image = pre_process_image(image)
    #crop
    image = wrap_crop_image(image)
    return image

