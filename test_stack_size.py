from shared import test_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop
import os

training_name = "stack_size"
image_size = (46, 46)

model_file_path = "./training/{}/{}.h5".format(training_name, training_name)
classes_file_path = "./training/{}/{}-classes.txt".format(training_name, training_name)
test_folder = "./images/{}/validation/".format(training_name)

if os.path.exists(model_file_path) and os.path.exists(classes_file_path):
    f = open(classes_file_path, "r")
    data = f.read()
    classes = data.split(",")[:-1]
    num_classes = len(classes)

    model = Sequential([
        Conv2D(16, (3, 3), activation="relu", input_shape=(image_size[0], image_size[1], 3)),
        MaxPool2D(2, 2),

        Conv2D(32, (3, 3), activation="relu"),
        MaxPool2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPool2D(2, 2),

        Flatten(),

        Dense(512, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer=RMSprop(learning_rate=0.001),
        metrics=["accuracy"]
    )

    model.load_weights(model_file_path)

    test_model(
        model=model,
        test_folder=test_folder,
        image_size=image_size,
        classes=classes
    )
else:
    print("No model named {} available".format(training_name))
