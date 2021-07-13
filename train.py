import os
import pathlib
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from test_model import test_model
import datetime

output_training_dir = "./training"


def fix_crash():
    # Tensorflow crashes and burns on a video card if this doesn't exist...
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def train_until_threshold(training_name, train_dir_path, image_size, test_folder, expected_accuracy=0.9, batch_size=16,
                          start_epochs=100, epoch_step=100, new_training=False):
    model_file_path, classes_file_path = train(training_name, train_dir_path, image_size, batch_size, start_epochs,
                                               new_training)
    success, success_rate = test_model(model_file_path, test_folder, classes_file_path)

    good_enough = success or success_rate >= expected_accuracy

    while not good_enough:
        model_file_path, classes_file_path = train(training_name, train_dir_path, image_size, batch_size, epoch_step, False)
        success, success_rate = test_model(model_file_path, test_folder, classes_file_path)

        if success or success_rate >= expected_accuracy:
            return model_file_path, classes_file_path


def train(training_name, train_dir_path, image_size, batch_size=16, epochs=100, new_training=False):
    data_dir = pathlib.Path(train_dir_path)

    img_width = image_size[0]
    img_height = image_size[1]

    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomTranslation(0.1, 0.1),
        layers.GaussianNoise(0.1)
    ])

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

    if not os.path.exists(output_training_dir):
        os.makedirs(output_training_dir)

    models_dir = '{}/{}/'.format(output_training_dir, training_name)

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    classes_file_path = "{}/{}/{}_classes.txt".format(output_training_dir, training_name, training_name)

    with open(classes_file_path, 'w') as f:
        f.write(','.join(class_names))

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    train_ds = train_ds.shuffle(100)
    num_classes = len(class_names)

    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(8, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    model_file_path = './{}/{}/{}.hdf5'.format(output_training_dir, training_name, training_name)

    if os.path.isfile(model_file_path) and not new_training:
        model.load_weights(model_file_path)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[tensorboard_callback]
    )

    model.save(model_file_path)
    return model_file_path, classes_file_path
