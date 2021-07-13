import os
from PIL import Image
import cv2

images_dir = "./images/item_links/training/"
size = (105, 145)

for folder in os.listdir(images_dir):
    items_folder = "{}/{}".format(images_dir, folder)

    for file_name in os.listdir(items_folder):
        path = ("{}/{}".format(items_folder, file_name))
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, size, cv2.INTER_AREA)
        cv2.imwrite(path, image)
