import os
import shutil

data_dir = "./images/currency_type/training/"

for currency_name in os.listdir(data_dir):
    for file_name in os.listdir(data_dir + currency_name):
        if currency_name == "empty":
            shutil.copy("{}{}/{}".format(data_dir, currency_name, file_name),
                        "./images/stack_size/training/{}/{}".format("empty", file_name))
        else:
            stack_size = file_name.split('.')[0]
            shutil.copy("{}{}/{}".format(data_dir, currency_name, file_name),
                        "./images/stack_size/training/{}/{}.png".format(stack_size, currency_name))
