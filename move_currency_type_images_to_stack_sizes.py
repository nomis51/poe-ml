import os
import shutil

currency_name = "chromatic_orb"
dir = "./images/currency_types/training/{}/".format(currency_name)

for file_name in os.listdir(dir):
    stack_size = file_name.split('.')[0]
    shutil.copy("{}{}".format(dir, file_name),
                "./images/stack_sizes/training/{}/{}.png".format(stack_size, currency_name))
