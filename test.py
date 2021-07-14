import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as Image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Sequential
import numpy as np
import os
import datetime
import cv2
from shared import test_model, process_images

training_name = "currency_type"
scale = 1 / 255
image_size = (46, 46)
batch_size = 16
epochs = 70
steps_per_epoch = 3
folder = "./images/{}/".format(training_name)
training_folder = "{}/training/".format(folder)
validation_folder = training_folder.replace("training", "validation")
original_folder = training_folder.replace("training", "original")
test_folder = validation_folder
classes = os.listdir(training_folder)
num_classes = len(classes)
log_dir = "logs/fit/{}/{}".format(training_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

train = ImageDataGenerator(rescale=scale)
validation = ImageDataGenerator(rescale=scale)

train_dataset = train.flow_from_directory(
    training_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical"
)
validation_dataset = train.flow_from_directory(
    validation_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical"
)

print("Class indices: ", validation_dataset.class_indices)

model = Sequential([
    Conv2D(16, (3, 3), activation="relu", input_shape=(46, 46, 3)),
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
    optimizer=RMSprop(lr=0.001),
    metrics=["accuracy"]
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[
        tensorboard_callback
    ],
    validation_data=validation_dataset
)

test_model(
    model=model,
    test_folder=test_folder,
    classes=classes,
    image_size=image_size
)
