import sys
from os import listdir
import random
from PySide6.QtCore import QTimer
from PySide6.QtGui import QPixmap, Qt
from PySide6.QtWidgets import QLabel

import model
from PySide6 import QtCore, QtWidgets, QtGui


class MyWidget(QtWidgets.QWidget):
    def __init__(self, cnn_model):
        super().__init__()
        self.main_layout = QtWidgets.QVBoxLayout(self)

        self.model = cnn_model
        self.test_image_path = "dataset/manual_test/"
        self.images = listdir(self.test_image_path)

        self.image_counter = 0
        self.image_height = 500
        self.image_label = QLabel()
        self.setup_ui()
        self.set_image()

        # Timer setup
        self.image_change_timer = QTimer(self)
        self.image_change_timer.timeout.connect(self.change_image)
        self.image_change_timer.setInterval(2500)
        self.image_change_timer.start()


    def setup_ui(self):
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.setContentsMargins(0, 20, 0, 20)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setContentsMargins(0, 20, 0, 20)
        self.image_label.setMinimumHeight(250)
        self.main_layout.addWidget(self.image_label)

        animal_1_label = self.class_prediction_label("Cat")
        animal_2_label = self.class_prediction_label("Cow")
        animal_3_label = self.class_prediction_label("Sheep")

        self.main_layout.setContentsMargins(10, 50, 10, 50)
        self.main_layout.setContentsMargins(10, 50, 10, 50)
        self.main_layout.setContentsMargins(10, 50, 10, 50)

        self.main_layout.addLayout(animal_1_label)
        self.main_layout.addLayout(animal_2_label)
        self.main_layout.addLayout(animal_3_label)

    def class_prediction_label(self, classname):
        layout = QtWidgets.QHBoxLayout()
        label = QLabel(classname)
        label.setFixedWidth(75)
        progress_bar = QtWidgets.QProgressBar(self)
        progress_bar.setGeometry(50, 100, 150, 30)
        progress_bar.setFixedWidth(550)
        progress_bar.setValue(0)

        layout.addWidget(label)
        layout.addWidget(progress_bar)
        return layout

    def set_image(self):
        # Get predictions
        predictions = model.predict_single(self.images[self.image_counter])
        # Display results
        self.set_progress_bars(predictions)
        pixmap = QPixmap(self.test_image_path + self.images[self.image_counter])
        pixmap = pixmap.scaledToHeight(self.image_height)
        self.image_label.setPixmap(pixmap)
        self.image_counter += 1

    def change_image(self):
        # Exit program if finished
        if self.image_counter >= len(self.images):
            self.image_change_timer.stop()
            sys.exit()
        # Get predictions
        predictions = model.predict_single(self.images[self.image_counter])
        # Display results
        self.set_progress_bars(predictions)
        pixmap = QPixmap(self.test_image_path + self.images[self.image_counter])
        pixmap = pixmap.scaledToWidth(self.image_height)
        self.image_label.setPixmap(pixmap)
        self.image_counter += 1

    def set_progress_bars(self, predictions):
        progress_bars = self.findChildren(QtWidgets.QProgressBar)
        predictions = predictions * 100
        progress_bars[0].setValue(round(predictions[0]))
        progress_bars[1].setValue(round(predictions[1]))
        progress_bars[2].setValue(round(predictions[2]))
        # progress_bars[3].setValue(round(predictions[3]))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    model = model.Model()

    widget = MyWidget(model)

    widget.resize(1000, 800)
    widget.show()

    sys.exit(app.exec())
