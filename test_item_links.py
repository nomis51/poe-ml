import os

from test_model import test_model

test_folder = "./images/item_links/training/"
model_file_path = "./training/item_links/item_links.hdf5"
classes_file_path = "./training/item_links/item_links_classes.txt"

success, success_rate = test_model(
    model_file_path, test_folder, classes_file_path)
