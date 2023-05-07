import sys  # to access the system
from os import listdir
from os.path import isfile, join, isdir
import tensorflow as tf
import numpy as np
from keras import utils
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

img_width = 128
img_height = 128
batch_size = 32


class Model:
    def __init__(self):
        # Allow memory growth for the GPU
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self.test_images_folder_path = "dataset/manual_test_scaled"
        self.load_model_path = "saved_models/model-83-82-70"
        self.input_shape = (img_width, img_height, 3)
        self.classes = {0: "Cat", 1: "Cow", 2: "Sheep"}
        # Preprocessing dataset
        self.train_data_gen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            channel_shift_range=0.2,
            rotation_range=20,
        )

        self.training_set = self.train_data_gen.flow_from_directory(
            'dataset/training_set_scaled',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical'
        )

        self.test_set_data_gen = ImageDataGenerator()

        self.test_set = self.test_set_data_gen.flow_from_directory(
            'dataset/test_set_scaled',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical'
        )

        if not isfile(self.load_model_path):
            self.model = self.construct_cnn()
        else:
            self.model = self.load_model()

    def construct_cnn(self):
        cnn = tf.keras.models.Sequential()
        cnn.add(tf.keras.layers.Input(shape=self.input_shape))
        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=1, strides=1))

        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        cnn.add(tf.keras.layers.Flatten())

        cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
        cnn.add(tf.keras.layers.Dropout(0.2))
        cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
        cnn.add(tf.keras.layers.Dense(units=3, activation='softmax'))

        cnn.compile(optimizer="adam",
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        # Save models per epoch
        model_path = "saved_models/" + "CNN" + "/Epoch{epoch:02d}-L{loss:.2f}-A{accuracy:.2f}-VL{val_loss:.2f}-VA{val_accuracy:.2f}.hdf5"
        model_save_callback = tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=False, monitor='loss')

        cnn.fit(x=self.training_set, validation_data=self.test_set, epochs=25, callbacks=[model_save_callback])
        cnn.save('saved_models/model', save_format="h5")

        return cnn

    def load_model(self):
        return tf.keras.models.load_model(self.load_model_path)

    def predict_single(self, img_name):
        img_path = self.test_images_folder_path + "/" + img_name

        test_image = load_img(img_path, target_size=(img_width, img_height))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = self.model.predict(test_image)[0]
        item_index = np.where(result == 1)

        print("--- ---")
        print("Image name: ", img_name)
        if len(item_index[0]):
            print("Guess: ", self.classes[item_index[0][0]])
        for index, prob in enumerate(result):
            print(self.classes[index], ": ", prob)

        return result

    def predict_single_class(self, vehicle_type, img):
        img_path = "dataset/test_set_scaled/" + vehicle_type + "/" + img
        test_image = load_img(img_path, target_size=(img_width, img_height))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = self.model.predict(test_image, verbose=0)[0]

        return np.where(result == max(result))[0][0]

    def get_per_class_accuracy(self, classes):
        print("\n")
        score_per_class = {}
        score = 0
        img_count = 0
        for index, type_directory in enumerate(classes):
            print(type_directory + " start")
            for img in listdir("dataset/test_set_scaled/" + type_directory + "/"):
                print(self.predict_single_class(type_directory, img), end=" ")
                if self.predict_single_class(type_directory, img) == index:
                    score += 1
                img_count += 1
                if img_count % 115 == 0:
                    print()
            score_per_class[type_directory] = score * 100 / img_count
            score = 0
            img_count = 0
            print("\n" + type_directory, " done")

        print("\n--- Per class accuracy -- \n")
        for key, value in score_per_class.items():
            print(key, ": ", value, "%")

