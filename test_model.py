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


def test_model(model_file_path, test_folder, classes_file_path):
    print("Testing model: " + model_file_path + " ...")
    model = tf.keras.models.load_model(model_file_path)
    class_names = []
    images = []

    with open(classes_file_path, "r") as file:
        class_names = file.read().split(",")

    for class_name in class_names:
        for file_name in os.listdir(test_folder + class_name):
            images.append([test_folder + class_name +
                          "/" + file_name, class_name])

    nb_fails = 0
    nb_success = 0

    for test in images:
        image = Image.open(test[0]).convert("RGB")
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        confidence = np.max(score)
        result = class_names[np.argmax(score)]

        if test[1] != result:
            nb_fails = nb_fails + 1
            print("BAD {} : {} -> {}".format(test[0], test[1], result))
        else:
            nb_success = nb_success + 1

    success_rate = 100 * (nb_success / (nb_success + nb_fails))

    print("Fails: {}. Success: {}".format(nb_fails, nb_success))
    print("Success rate: {}%".format(success_rate))

    return (False, success_rate / 100) if nb_fails > 0 else (True, 1)
