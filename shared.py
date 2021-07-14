import os
import cv2
from tensorflow.keras.preprocessing import image as Image
import numpy as np
import tensorflow as tf


def test_model(model, test_folder, image_size, classes):
    print()
    print("Testing model...")

    nb_success = 0
    nb_fail = 0

    for label in os.listdir(test_folder):
        current_path = "{}{}".format(test_folder, label)

        for file_name in os.listdir(current_path):
            image_path = "{}/{}".format(current_path, file_name)
            img = Image.load_img(image_path, target_size=image_size)
            x = Image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])

            predictions = model.predict(images)
            score = tf.nn.softmax(predictions[0])
            confidence = np.max(score)
            result = classes[np.argmax(score)]

            if result != label:
                nb_fail = nb_fail + 1
                print("Expected -> Prediction : {} -> {}. {}".format(label, result, image_path))
            else:
                nb_success = nb_success + 1

    print("{} success. {} fail{}.".format(nb_success, nb_fail, "s" if nb_fail > 1 else ""))
    print("Success rate: {}%".format(100 * (nb_success / (nb_success + nb_fail))))


def process_images(images_dir, callbacks):
    original_folder = "{}original/".format(images_dir)
    training_folder = original_folder.replace("original", "training")

    if not os.path.exists(training_folder):
        os.mkdir(training_folder)

    for current_class in os.listdir(original_folder):
        original_current_class_folder = "{}{}".format(original_folder, current_class)
        training_current_class_folder = "{}{}".format(training_folder, current_class)

        if not os.path.exists(training_current_class_folder):
            os.mkdir(training_current_class_folder)

        for file_name in os.listdir(original_current_class_folder):
            original_path = "{}/{}".format(original_current_class_folder, file_name)
            training_path = "{}/{}".format(training_current_class_folder, file_name)

            image = cv2.imread(original_path, cv2.IMREAD_UNCHANGED)

            for callback in callbacks:
                image = callback(image)

            cv2.imwrite(training_path, image)
