import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2  # Used in function 'load_datasets'
import glob  # Used in function 'load_datasets'

# Importing dataset
# Paths to individual folders containing images regarding classes
malignant_folder_path = r"C:\Users\chris\Desktop\Studium\PhD\Courses\Spring 2020\COSC 525 - Deep Learning\DeepLearning_Project5\dataset\malignant\\"
benign_folder_path = r"C:\Users\chris\Desktop\Studium\PhD\Courses\Spring 2020\COSC 525 - Deep Learning\DeepLearning_Project5\dataset\benign\\"
normal_folder_path = r"C:\Users\chris\Desktop\Studium\PhD\Courses\Spring 2020\COSC 525 - Deep Learning\DeepLearning_Project5\dataset\normal\\"
paths = [malignant_folder_path, benign_folder_path, normal_folder_path]


def resize_images(resized_width, resized_height, class_paths):
    size = resized_width, resized_height
    for folder_path in class_paths:
        pgm_image_list = glob.glob(folder_path + '*.jpg')
        for filename in pgm_image_list:
            img = Image.open(filename)
            img.thumbnail(size)
            img.save(filename[:-4]+"_resize" + '.jpg')


def load_datasets(class_paths):
    """
    Loading and storing information real mammography images in arrays
    Categories: normal, malignant, benign
    :param class_paths: Array containing three file paths (normal, malignant, benign)
    :return: three n*p (p=3) matrices containing information about the three different classes
    """
    datasets = []
    for class_path in class_paths:
        dataset = []
        for image in glob.glob(class_path + "*_resize.jpg"):
            print(len(image))
            dataset.append(cv2.imread(image))
        datasets.append(dataset)
    return datasets[0], datasets[1], datasets[2]


def create_X_y(malignant_data, benign_data, normal_data):
    one_hot_encoding = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    X = malignant_data + benign_data + normal_data
    y = len(malignant_data) * [one_hot_encoding[0]] + len(benign_data) * [one_hot_encoding[1]] + len(normal_data) * [
        one_hot_encoding[2]]
    return X, y


def preprocess_data():
    print("Loading Dataset...")

    X_malignant, X_benign, X_normal = load_datasets(class_paths=paths)
    X, y = create_X_y(X_malignant, X_benign, X_normal)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.254658, random_state=42)
    X_train = np.array(X_train - 127.5) / 127.5
    X_test = np.array(X_test - 127.5) / 127.5

    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    print("Finished loading!")

    return X_train, X_test, y_train, y_test
