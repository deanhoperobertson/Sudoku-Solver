[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

# Sudoku-Solver

This program extracts, proceses and solves Sudoku images taken from local newspapers in London. Here we use a colectiong of image preprocese before applying a training convolutional neural network (CNN) to extract the digits from the individual cells. The CNN was training on the MNIST dataset on handwritten digits.

![Example](https://github.com/deanhoperobertson/Sudoku-Solver/blob/main/Images/Easy.jpg?raw=true)

## How to Run
```    
python cnn.py (to train model)
python sudoku.py
```

## Package Dependencies
- Python 3.7
- OpenCV
- Tensorflow
- Keras
