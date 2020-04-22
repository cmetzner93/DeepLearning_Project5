# COSC 525 - Deep Learning
# Assignment: Final Project
# Title: Improving mammography classification with conditional DCGAN generated images.
# Team members: Christoph Metzner, Anna-Maria Nau
# Date: 04/21/2020

# Imported Libaries
import tensorflow as tf
import keras
import numpy as np
import sys

# Import script to preprocess the data
# noinspection PyUnresolvedReferences
from Data_Preprocessing import preprocess_data


# Importing dataset
# Paths to individual folders containing images regarding classes
malignant_folder_path = r"C:\Users\chris\Desktop\Studium\PhD\Courses\Spring 2020\COSC 525 - Deep Learning\DeepLearning_FinalProject\Dataset\Malignant\\"
benign_folder_path = r"C:\Users\chris\Desktop\Studium\PhD\Courses\Spring 2020\COSC 525 - Deep Learning\DeepLearning_FinalProject\Dataset\Benign\\"
normal_folder_path = r"C:\Users\chris\Desktop\Studium\PhD\Courses\Spring 2020\COSC 525 - Deep Learning\DeepLearning_FinalProject\Dataset\Normal\\"
paths = [malignant_folder_path, benign_folder_path, normal_folder_path]


def main(argv=None):
    X_train, X_test, y_train, y_test = preprocess_data(class_paths=paths)
    print(X_train)


if __name__ == "__main__":
    main(sys.argv)
