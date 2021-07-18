import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as Image
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop

training_name = "stack_size"
image_size = (46, 46)
model_file_path = "./training/{}/{}.h5".format(training_name, training_name)
classes_file_path = "./training/{}/{}-classes.txt".format(training_name, training_name)
test_folder = "./images/{}/validation/".format(training_name)
image_path = "./images/tests/chaos10_1.png"

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

EMPTY_IMAGE_THRESHOLD = 0.15
empty_file_path = "./images/currency_type/validation/empty/empty_1.png"
empty_image = cv2.imread(empty_file_path, cv2.IMREAD_COLOR)


test_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
test_image = cv2.resize(test_image, image_size, interpolation=cv2.INTER_AREA)
print(test_image.shape, empty_image.shape)
match = cv2.matchTemplate(empty_image, test_image, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)

if max_val >= EMPTY_IMAGE_THRESHOLD:
    print("It's an empty image: {}%".format(max_val))

img = Image.load_img(image_path, target_size=image_size)
x = Image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])

predictions = model.predict(images)
score = tf.nn.softmax(predictions[0])
# confidence = np.max(score)
result = classes[np.argmax(score)]

print("Result: ", result)
