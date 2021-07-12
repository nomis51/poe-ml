import os
import tensorflow as tf
from PIL import Image
import numpy as np


def test_model_once(model_file_path, image_file_path, size, classes_file_path, expected_result):
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
    score = tf.nn.softmax(predictions[0])
    confidence = np.max(score)
    result = class_names[np.argmax(score)]

    if expected_result != result:
        print("BAD {} : {} -> {}".format(image_file_path, expected_result, result))
    else:
        print("GOOD {} : {} -> {}".format(image_file_path, expected_result, result))


def test_model(model_file_path, test_folder, classes_file_path):
    model = tf.keras.models.load_model(model_file_path)
    class_names = []
    images = []

    with open(classes_file_path, "r") as file:
        class_names = file.read().split(",")

    for class_name in class_names:
        for file_name in os.listdir(test_folder + class_name):
            images.append([test_folder + class_name + "/" + file_name, class_name])

    nbFails = 0
    nbSuccess = 0

    for test in images:
        image = Image.open(test[0]).convert("RGB")
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        confidence = np.max(score)
        result = class_names[np.argmax(score)]

        if test[1] != result:
            nbFails = nbFails + 1
            print("BAD {} : {} -> {}".format(test[0], test[1], result))
        else:
            nbSuccess = nbSuccess + 1
            print("GOOD {} : {} -> {}".format(test[0], test[1], result))

    success_rate = 100 * (nbSuccess / (nbSuccess + nbFails))

    print("Fails: {}. Success: {}".format(nbFails, nbSuccess))
    print("Success rate: {}%".format(success_rate))

    return (False, success_rate / 100) if nbFails > 0 else (True, 1)
