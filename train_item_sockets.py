import tensorflow as tf
import os
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from shared import test_model, AccuracyThresholdCallback

training_name = "item_sockets"
scale = 1 / 255
image_size = (200, 200)
batch_size = 4
epochs = 100
steps_per_epoch = 3
load_existing_model = False
accuracy_threshold = 1.0
loss_threshold = 0.009
test_empty_image = False

folder = "./images/{}/".format(training_name)
training_folder = "{}/training/".format(folder)
validation_folder = training_folder.replace("training", "validation")
original_folder = training_folder.replace("training", "original")
test_folder = validation_folder
classes = os.listdir(training_folder)
num_classes = len(classes)
log_dir = "logs/fit/{}/{}".format(training_name,
                                  datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
output_path = "./training/{}/".format(training_name)
model_output_path = "{}{}.h5".format(output_path, training_name)
classes_output_path = "{}{}-classes.txt".format(output_path, training_name)

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
    Conv2D(16, (3, 3), activation="relu", input_shape=(
        image_size[0], image_size[1], 3)),
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

model.summary()

if load_existing_model and os.path.exists(model_output_path):
    model.load_weights(model_output_path)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

accuracy_threshold_callback = AccuracyThresholdCallback(
    accuracy_threshold=accuracy_threshold,
    loss_threshold=loss_threshold
)

model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[
        tensorboard_callback,
        accuracy_threshold_callback
    ],
    validation_data=validation_dataset
)

model.save(model_output_path)

f = open(classes_output_path, "w")
for c in classes:
    f.write(c + ",")
f.close()

test_model(
    model=model,
    test_folder=test_folder,
    classes=classes,
    image_size=image_size,
    test_empty_image=test_empty_image
)
