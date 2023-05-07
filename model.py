import sys  # to access the system
from os import listdir
from os.path import isfile, join, isdir
import tensorflow as tf
import numpy as np
from keras import utils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array

img_width = 64
img_height = 64
batch_size = 64


class Model:
    def __init__(self):
        self.test_images_folder_path = "dataset/manual_test"
        # Preprocessing dataset
        self.train_data_gen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rotation_range=20,
        )

        self.training_set = self.train_data_gen.flow_from_directory(
            'dataset/training_set_scaled',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical'
        )
        self.test_set = self.train_data_gen.flow_from_directory(
            'dataset/test_set_scaled',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical'
        )
        self.classes = {}
        for key, value in self.training_set.class_indices.items():
            self.classes[value] = (key[:-1] if key != "buses" else key[:-2]).capitalize()

        if not isdir("saved_models/model") or not listdir("saved_models/model"):
            self.model = self.construct_cnn()
        else:
            self.model = self.load_model()

        # print(self.model.summary())
        print(self.training_set.class_indices)
        # self.model.evaluate(self.test_set)

        print(self.get_per_class_accuracy(["cats","cows","sheeps"]))

    def construct_cnn(self):
        cnn = tf.keras.models.Sequential()

        cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=1, strides=1))

        cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        cnn.add(tf.keras.layers.Flatten())
        cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))
        cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
        cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
        cnn.add(tf.keras.layers.Dense(units=3, activation='softmax'))

        cnn.compile(optimizer="adam",
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        cnn.fit(x=self.training_set, validation_data=self.test_set, epochs=40)
        cnn.save('saved_models/model')

        return cnn

    def load_model(self):
        return tf.keras.models.load_model("saved_models/model")

    def predict_single(self, img_name):
        img_path = self.test_images_folder_path + "/" + img_name

        test_image = utils.load_img(img_path, target_size=(img_width, img_height))
        test_image = utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        predictions = (tf.nn.softmax(self.model.predict(test_image)[0])).numpy()
        result = self.model.predict(test_image)[0]
        item_index = np.where(result == 1)

        print("--- ---")
        print("Image name: ", img_name)
        print(predictions)
        if len(item_index[0]):
            print("Guess: ", self.classes[item_index[0][0]])
        for index, prob in enumerate(result):
            print(self.classes[index], ": ", prob)

        return predictions

    def predict_single_class(self, vehicle_type, img):
        img_path = "dataset/test_set/" + vehicle_type + "/" + img
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
            for img in listdir("dataset/test_set/" + type_directory + "/"):
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

    def predict_multiple(self):
        test_images = listdir(self.test_images_folder_path)

        for img in test_images:
            self.predict_single(img)
