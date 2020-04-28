from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from cDCGAN_data_prep import preprocess_data
from tensorflow.keras.optimizers import SGD


# light-weight version of VGG16
def create_model():
    model = Sequential()

    # convolutional layers
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # fully connected layer
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    # compile model
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model


# load data
X_train, X_test, y_train, y_test = preprocess_data()

# create model
model = create_model()

# fit model
history = model.fit(X_train, y_train, epochs=5, batch_size=10, verbose = 1)

