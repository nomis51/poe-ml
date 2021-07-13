import tensorflow as tf
from PIL import Image
import numpy as np

model_file_path = "./training/currency_type/currency_type.hdf5"
classes_file_path = "./training/currency_type/currency_type_classes.txt"
size = (46, 46)
image_file_path = "./images/tests/chaos4.png"

model = tf.keras.models.load_model(model_file_path)

class_names = []
images = []

with open(classes_file_path, "r") as file:
    class_names = file.read().split(",")

image = Image.open(image_file_path).convert("RGB")
image = tf.image.resize(image, [size[0], size[1]])
img_array = tf.keras.preprocessing.image.img_to_array(image)
img_array = tf.expand_dims(img_array, 0)
predictions = model.predict(img_array)

for prediction in predictions:
    print("Prediction: ", prediction)
    score = tf.nn.softmax(prediction)
    confidence = np.max(score)
    result = class_names[np.argmax(score)]
    print("Score: ", score)
    print("Confidence: ", confidence)
    print("Result: ", result)
