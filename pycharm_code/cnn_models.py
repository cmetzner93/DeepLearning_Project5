import time
import cv2
import glob
import sys
import numpy as np
from pickle import dump
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split


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
            dataset.append(cv2.imread(image))
        datasets.append(dataset)
    return datasets[0], datasets[1], datasets[2]


# function that splits data in X and y
def create_X_y(malignant_data, benign_data, normal_data):
    one_hot_encoding = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    X = malignant_data + benign_data + normal_data
    y = len(malignant_data) * [one_hot_encoding[0]] + len(benign_data) * [one_hot_encoding[1]] + len(normal_data) * [
        one_hot_encoding[2]]
    return X, y

# function that creates train and test sets
def preprocess_data(paths):
    print("Loading Dataset...")

    X_malignant, X_benign, X_normal = load_datasets(class_paths=paths)
    X, y = create_X_y(X_malignant, X_benign, X_normal)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.254658, random_state=42)
    X_train = np.array(X_train) / 255
    X_test = np.array(X_test) / 255

    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    print("Finished loading!")

    return X_train, X_test, y_train, y_test


# 3-layer CNN
def cnn_3(img_height, img_weight, channels, num_classes):
    model = Sequential()

    # convolutional layers
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                            input_shape=(img_height, img_weight, channels)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # fully connected layer
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # compile model
    sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model


# 4-layer CNN
def cnn_4(img_height, img_weight, channels, num_classes):
    model = Sequential()

    # convolutional layers
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                            input_shape=(img_height, img_weight, channels)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # fully connected layer
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # compile model
    sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def train_and_predict(model, X_train, y_train, X_test, y_test, epochs, batch_size, model_name):
    # call for taking time stamps for each epoch
    time_callback_train = TimeHistory()

    # fit model
    # model is not trained on validation data, data is used to evaluate loss and any model metric after each epoch
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                        callbacks=[time_callback_train])

    # save model for later use
    model.save('%s.h5' % (model_name))
    # save model history
    with open('%s.pkl' % (model_name), 'wb') as file:
        dump(history.history, file)

    # store time stamps per epoch in variable
    times_train = time_callback_train.times
    print()
    # print("Time per epoch: \n ", times_train)
    print('Total train time: %.3f seconds' % (sum(times_train)))
    print()
    print()

    # Evaluate model using testing dataset
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss: {} and Test accuracy: {}'.format(test_loss, test_acc))

    # plots
    # epochs vs. accuracy
    plt.figure()
    plt.ylim(0.4, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Epochs vs. Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig('epoch_acc_%s.png' % (model_name))

    # epochs vs. loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Epochs vs. Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()
    plt.savefig('epoch_loss_%s.png' % (model_name))



# Driver code main()
def main(argv=None):
    # build 3-layer cnn using 128 x 128 pixels image data
    if argv[1] == '128' and argv[2] == '3':
        malignant_folder_path = 'data/malignant_128/'
        benign_folder_path = 'data/benign_128/'
        normal_folder_path = 'data/normal_128/'
        paths = [malignant_folder_path, benign_folder_path, normal_folder_path]

        # load data
        X_train, X_test, y_train, y_test = preprocess_data(paths)

        # create and fit model, then predict test data, create plots
        model = cnn_3(img_height=128, img_weight=128, channels=3, num_classes=3)
        train_and_predict(model, X_train, y_train, X_test, y_test, epochs=200, batch_size=32, model_name='cnn_3_128')

    # build 4-layer cnn using 128 x 128 pixels image data
    elif argv[1] == '128' and argv[2] == '4':
        malignant_folder_path = 'data/malignant_128/'
        benign_folder_path = 'data/benign_128/'
        normal_folder_path = 'data/normal_128/'
        paths = [malignant_folder_path, benign_folder_path, normal_folder_path]

        # load data
        X_train, X_test, y_train, y_test = preprocess_data(paths)

        # create and fit model, then predict test data, create plots
        model = cnn_4(img_height=128, img_weight=128, channels=3, num_classes=3)
        train_and_predict(model, X_train, y_train, X_test, y_test, epochs=200, batch_size=32, model_name='cnn_4_128')

    # build 3-layer cnn using 256 x 256 pixels image data
    elif argv[1] == '256' and argv[2] == '3':
        malignant_folder_path = 'data/malignant_256/'
        benign_folder_path = 'data/benign_256/'
        normal_folder_path = 'data/normal_256/'
        paths = [malignant_folder_path, benign_folder_path, normal_folder_path]

        # load data
        X_train, X_test, y_train, y_test = preprocess_data(paths)

        # create and fit model, then predict test data, create plots
        model = cnn_3(img_height=256, img_weight=256, channels=3, num_classes=3)
        train_and_predict(model, X_train, y_train, X_test, y_test, epochs=200, batch_size=32, model_name='cnn_3_256')

    # build 4-layer cnn using 256 x 256 pixels image data
    elif argv[1] == '256' and argv[2] == '4':
        malignant_folder_path = 'data/malignant_256/'
        benign_folder_path = 'data/benign_256/'
        normal_folder_path = 'data/normal_256/'
        paths = [malignant_folder_path, benign_folder_path, normal_folder_path]

        # load data
        X_train, X_test, y_train, y_test = preprocess_data(paths)

        # create and fit model, then predict test data, create plots
        model = cnn_4(img_height=256, img_weight=256, channels=3, num_classes=3)
        train_and_predict(model, X_train, y_train, X_test, y_test, epochs=200, batch_size=32, model_name='cnn_4_256')


if __name__ == '__main__':
    main(sys.argv)
