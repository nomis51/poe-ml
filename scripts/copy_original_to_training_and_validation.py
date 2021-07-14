import os
import shutil


def copy_data():
    training_names = ["currency_type", "item_links", "stack_size", "item_sockets"]

    for training_name in training_names:
        folder = "./images/{}/".format(training_name)
        training_folder = "{}training/".format(folder)
        validation_folder = training_folder.replace("training", "validation")
        original_folder = training_folder.replace("training", "original")

        if not os.path.exists(training_folder):
            os.mkdir(training_folder)

        if not os.path.exists(validation_folder):
            os.mkdir(validation_folder)

        for classe in os.listdir(original_folder):
            original_classe_folder = "{}{}/".format(original_folder, classe)
            training_classe_folder = "{}{}/".format(validation_folder, classe)
            validation_classe_folder = "{}{}/".format(training_folder, classe)

            if not os.path.exists(training_classe_folder):
                os.mkdir(training_classe_folder)

            if not os.path.exists(validation_classe_folder):
                os.mkdir(validation_classe_folder)

            for file_name in os.listdir(original_classe_folder):
                shutil.copyfile("{}{}".format(original_classe_folder, file_name),
                                "{}{}".format(training_classe_folder, file_name))
                shutil.copyfile("{}{}".format(original_classe_folder, file_name),
                                "{}{}".format(validation_classe_folder, file_name))

copy_data()