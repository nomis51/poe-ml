import os

from test_model import test_model

test_folder = "./images/currency_types/training/"
model_file_path = "./training/currency_type/currency_type.hdf5"
classes_file_path = "./training/currency_type/currency_type_classes.txt"

success, success_rate = test_model(model_file_path, test_folder, classes_file_path)
