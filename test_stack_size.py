import os

from test_model import test_model

test_folder = "./images/stack_sizes/training/"
model_file_path = "./training/stack_size/stack_size.hdf5"
classes_file_path = "./training/stack_size/stack_size_classes.txt"

success, success_rate = test_model(model_file_path, test_folder, classes_file_path)
