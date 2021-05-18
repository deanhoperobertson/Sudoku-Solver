#apply cnn model to extract digits.
from keras.models import model_from_json


#LOAD model architecture
json_file = open('models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#LOAD model pre-trained weights
loaded_model.load_weights("models/model.h5")
print("Loaded saved model from disk.")
