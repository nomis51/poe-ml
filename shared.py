import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as Image


def test_model(model, test_folder, image_size, classes, test_empty_image = True):
    print()
    print("Testing model...")

    EMPTY_IMAGE_THRESHOLD = 0.15
    empty_file_path = "./images/currency_type/validation/empty/empty_1.png"
    empty_image = cv2.imread(empty_file_path, cv2.IMREAD_COLOR)

    nb_success = 0
    nb_fail = 0

    for label in os.listdir(test_folder):
        current_path = "{}{}".format(test_folder, label)

        for file_name in os.listdir(current_path):
            image_path = "{}/{}".format(current_path, file_name)

            if test_empty_image:
                test_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                test_image = cv2.resize(test_image, image_size, interpolation=cv2.INTER_AREA)
                match = cv2.matchTemplate(empty_image, test_image, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)

                if max_val >= EMPTY_IMAGE_THRESHOLD:
                    nb_success = nb_success + 1
                    print("It's an empty image: {}%".format(max_val))
                    continue

            img = Image.load_img(image_path, target_size=image_size)
            x = Image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])

            predictions = model.predict(images)
            score = tf.nn.softmax(predictions[0])
            # confidence = np.max(score)
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
