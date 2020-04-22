import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2  # Used in function 'load_datasets'
import glob  # Used in function 'load_datasets'


def load_datasets(class_paths):
    """
    Loading and storing information real mammography images in arrays
    Categories: normal, malignant, benign
    :param class_paths: Array containing three file paths (normal, malignant, benign)
    :return: three n*p (p=3) matrices containing information about the three different classes
    """
    print("Loading Dataset...")
    datasets = []
    for class_path in class_paths:
        dataset = []
        for image in glob.glob(class_path + "*.jpg"):
            dataset.append(cv2.imread(image))
        datasets.append(dataset)
    print("Finished loading!")
    return datasets[0], datasets[1], datasets[2]


def create_X_y(malignant_data, benign_data, normal_data):
    one_hot_encoding = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    X = malignant_data + benign_data + normal_data
    y = len(malignant_data)*[one_hot_encoding[0]]+len(benign_data)*[one_hot_encoding[1]]+len(normal_data)*[one_hot_encoding[2]]
    return X, y


def preprocess_data(class_paths):
    X_malignant, X_benign, X_normal = load_datasets(class_paths=class_paths)
    X, y = create_X_y(X_malignant, X_benign, X_normal)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train = np.array(X_train) / 255
    X_test = np.array(X_test) / 255

    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    return X_train, X_test, y_train, y_test