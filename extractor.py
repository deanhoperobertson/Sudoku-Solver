#apply cnn model to extract digits.
import cv2
import numpy as np
from keras.models import model_from_json

from preprocess import (
    create_grid,
    cut_from_rect,
    has_number,
    clean_number_cell,
    find_bounding_box)



#LOAD model architecture
json_file = open('models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#LOAD model pre-trained weights
loaded_model.load_weights("models/model.h5")
print("Loaded saved model from disk.")


def identify_number(image: np.ndarray) -> int:
    image_resize = cv2.resize(image, (28,28))
    image_resize_2 = image_resize.reshape(1,1,28,28).astype('float32')
    loaded_model_pred = loaded_model.predict_classes(image_resize_2 , verbose = 0)

    return loaded_model_pred[0]

def extract_sudoku(image: np.ndarray):
    '''
    '''
    image = image.copy()
    grid = create_grid(image)
    numbers = []
    count = 0

    for square in grid:
        print(count)
        image_cell = cut_from_rect(image, square)

        if has_number(image_cell) == False:
            numbers.append("NO")
        else:
            clean_image = clean_number_cell(image_cell)
            clean_image = cv2.resize(clean_image, (222*1, 225*1)) 
            cv2.imshow("Window", clean_image)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()

            digit = identify_number(clean_image)
            numbers.append(digit)
        count = count + 1

    return numbers

