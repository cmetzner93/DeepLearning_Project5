# COSC 525 - Deep Learning
# Assignment: Final Project
# Title: Improving mammography classification with conditional DCGAN generated images.
# Team members: Christoph Metzner, Anna-Maria Nau
# Date: 04/21/2020

# Imported Libaries
import sys
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import os

# Import script to preprocess the data
# noinspection PyUnresolvedReferences
# from Data_Preprocessing import preprocess_data
# from cDCGAN import generator_model
# from cDCGAN import discriminator_model
from sklearn.model_selection import train_test_split
import cv2  # Used in function 'load_datasets'
import glob  # Used in function 'load_datasets'

tf.keras.backend.clear_session()  # For easy reset of notebook state.

# Importing dataset
# Paths to individual folders containing images regarding classes
malignant_folder_path = r"C:\Users\chris\Desktop\Studium\PhD\Courses\Spring 2020\COSC 525 - Deep Learning\DeepLearning_FinalProject\Dataset\Malignant\\"
benign_folder_path = r"C:\Users\chris\Desktop\Studium\PhD\Courses\Spring 2020\COSC 525 - Deep Learning\DeepLearning_FinalProject\Dataset\Benign\\"
normal_folder_path = r"C:\Users\chris\Desktop\Studium\PhD\Courses\Spring 2020\COSC 525 - Deep Learning\DeepLearning_FinalProject\Dataset\Normal\\"
paths = [malignant_folder_path, benign_folder_path, normal_folder_path]


# Functions used for preprocessing the real image dataset
# load_datasets: load datasets from directory
# create_X_y: generate a complete dataset with images X and respective labels y
# preprocess_data: Split X and y into train and test and convert them into tensors type=float32
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
        for image in glob.glob(class_path + "*.jpg"):
            dataset.append(cv2.imread(image))
        datasets.append(dataset)
    return datasets[0], datasets[1], datasets[2]


def create_X_y(malignant_data, benign_data, normal_data):
    one_hot_encoding = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    X = malignant_data + benign_data + normal_data
    y = len(malignant_data) * [one_hot_encoding[0]] + len(benign_data) * [one_hot_encoding[1]] + len(normal_data) * [
        one_hot_encoding[2]]
    return X, y


def preprocess_data(class_paths):
    print("Loading Dataset...")

    X_malignant, X_benign, X_normal = load_datasets(class_paths=class_paths)
    X, y = create_X_y(X_malignant, X_benign, X_normal)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train = np.array(X_train) / 255
    X_test = np.array(X_test) / 255

    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    print("Finished loading!")

    return X_train, X_test, y_train, y_test


def generator_model():
    # Prepare noise input z
    input_z = tf.keras.layers.Input(shape=(100,))
    dense_z_1 = tf.keras.layers.Dense(1024 * 4 * 4)(input_z)
    act_z_1 = tf.keras.layers.LeakyReLU(alpha=0.2)(dense_z_1)
    bn_z_1 = tf.keras.layers.BatchNormalization(momentum=0.9)(act_z_1)
    reshape_z = tf.keras.layers.Reshape(target_shape=(4, 4, 1024), input_shape=(4 * 4 * 1024,))(bn_z_1)

    # prepare conditional (label) input c
    input_c = tf.keras.layers.Input(shape=(3,))
    dense_c_1 = tf.keras.layers.Dense(4 * 4 * 1)(input_c)
    act_c_1 = tf.keras.layers.LeakyReLU(alpha=0.2)(dense_c_1)
    bn_c_1 = tf.keras.layers.BatchNormalization(momentum=0.9)(act_c_1)
    reshape_c = tf.keras.layers.Reshape(target_shape=(4, 4, 1), input_shape=(4 * 4 * 1,))(bn_c_1)

    # concatenating noise z and label c
    concat_z_c = tf.keras.layers.Concatenate()([reshape_z, reshape_c])

    # Image generation
    # Upsampling to 8x8
    conv2D_1 = tf.keras.layers.Conv2DTranspose(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding='same')(
        concat_z_c)
    act_conv2D_1 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2D_1)
    bn_conv2D_1 = tf.keras.layers.BatchNormalization(momentum=0.9)(act_conv2D_1)

    # Upsampling to 16x16
    conv2D_2 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same')(
        bn_conv2D_1)
    act_conv2D_2 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2D_2)
    bn_conv2D_2 = tf.keras.layers.BatchNormalization(momentum=0.9)(act_conv2D_2)

    # Upsampling to 32x32
    conv2D_3 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same')(
        bn_conv2D_2)
    act_conv2D_3 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2D_3)
    bn_conv2D_3 = tf.keras.layers.BatchNormalization(momentum=0.9)(act_conv2D_3)

    # Upsampling to 64x64
    conv2D_4 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(
        bn_conv2D_3)
    act_conv2D_4 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2D_4)
    bn_conv2D_4 = tf.keras.layers.BatchNormalization(momentum=0.9)(act_conv2D_4)

    # Upsampling to 128x128
    conv2D_5 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(
        bn_conv2D_4)
    act_conv2D_5 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2D_5)
    bn_conv2D_5 = tf.keras.layers.BatchNormalization(momentum=0.9)(act_conv2D_5)

    # Upsampling to 256x256
    conv2D_6 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(
        bn_conv2D_5)
    act_conv2D_6 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2D_6)
    bn_conv2D_6 = tf.keras.layers.BatchNormalization(momentum=0.9)(act_conv2D_6)

    # Upsampling to 512x512
    conv2D_7 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same')(
        bn_conv2D_6)
    act_conv2D_7 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2D_7)
    bn_conv2D_7 = tf.keras.layers.BatchNormalization(momentum=0.9)(act_conv2D_7)

    # Upsampling to 1024x1024
    conv2D_8 = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=(2, 2), padding='same')(
        bn_conv2D_7)
    act_conv2D_8 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2D_8)
    bn_conv2D_8 = tf.keras.layers.BatchNormalization(momentum=0.9)(act_conv2D_8)

    # Output layer
    conv2D_9 = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), activation='tanh', padding='same')(
        bn_conv2D_8)

    # Model output
    model = tf.keras.models.Model(inputs=[input_z, input_c], outputs=conv2D_9)
    return model


def discriminator_model():
    # prepare conditional (label) input c
    input_c = tf.keras.layers.Input(shape=(3,))
    dense_c_1 = tf.keras.layers.Dense(1024 * 1024 * 1)(input_c)
    act_c_1 = tf.keras.layers.LeakyReLU(alpha=0.2)(dense_c_1)
    bn_c_1 = tf.keras.layers.BatchNormalization(momentum=0.9)(act_c_1)
    reshape_c = tf.keras.layers.Reshape(target_shape=(1024, 1024, 1), input_shape=(1024 * 1024 * 1,))(bn_c_1)

    # Get input images x: real p(x_r) or fake p(x_z)
    input_x = tf.keras.layers.Input(shape=(1024, 1024, 3))

    # Concatenate input c and image x
    concat_x_c = tf.keras.layers.Concatenate()([input_x, reshape_c])

    # Feature extraction for discriminating real from fake images
    # Begin feature extraction process
    # Downsampling: 16 feature maps
    conv2d_1 = tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', name='conv_512x512')(concat_x_c)
    act_conv2d_1 = tf.keras.layers.LeakyReLU(alpha=0.2, name='lReLU_512x512')(conv2d_1)
    dp_conv2d_1 = tf.keras.layers.Dropout(0.33, name='Dropout_512x512')(act_conv2d_1)

    # Downsampling: 32 feature maps
    conv2d_2 = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', name='conv_256x256')(dp_conv2d_1)
    act_conv2d_2 = tf.keras.layers.LeakyReLU(alpha=0.2, name='lReLU_256x256')(conv2d_2)
    dp_conv2d_2 = tf.keras.layers.Dropout(0.33, name='Dropout_256x256')(act_conv2d_2)

    # Downsampling: 64 feature maps
    conv2d_3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', name='conv_128x128')(dp_conv2d_2)
    act_conv2d_3 = tf.keras.layers.LeakyReLU(alpha=0.2, name='lReLU_128x128')(conv2d_3)
    dp_conv2d_3 = tf.keras.layers.Dropout(0.33, name='Dropout_128x128')(act_conv2d_3)

    # Downsampling: 128 feature maps
    conv2d_4 = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', name='conv_64x64')(dp_conv2d_3)
    act_conv2d_4 = tf.keras.layers.LeakyReLU(alpha=0.2, name='lReLU_64x64')(conv2d_4)
    dp_conv2d_4 = tf.keras.layers.Dropout(0.33, name='Dropout_64x64')(act_conv2d_4)

    # Downsampling: 256 feature maps
    conv2d_5 = tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', name='conv_32x32')(dp_conv2d_4)
    act_conv2d_5 = tf.keras.layers.LeakyReLU(alpha=0.2, name='lReLU_32x32')(conv2d_5)
    dp_conv2d_5 = tf.keras.layers.Dropout(0.33, name='Dropout_32x32')(act_conv2d_5)

    # Downsampling: 512 feature maps
    conv2d_6 = tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', name='conv_16x16')(dp_conv2d_5)
    act_conv2d_6 = tf.keras.layers.LeakyReLU(alpha=0.2, name='lReLU_16x16')(conv2d_6)
    dp_conv2d_6 = tf.keras.layers.Dropout(0.33, name='Dropout_16x16')(act_conv2d_6)

    # Downsampling: 512 feature maps
    conv2d_7 = tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', name='conv_8x8')(dp_conv2d_6)
    act_conv2d_7 = tf.keras.layers.LeakyReLU(alpha=0.2, name='lReLU_8x8')(conv2d_7)
    dp_conv2d_7 = tf.keras.layers.Dropout(0.33, name='Dropout_8x8')(act_conv2d_7)

    # Downsampling: 512 feature maps
    conv2d_8 = tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', name='conv_4x4')(dp_conv2d_7)
    act_conv2d_8 = tf.keras.layers.LeakyReLU(alpha=0.2, name='lReLU_4x4')(conv2d_8)
    dp_conv2d_8 = tf.keras.layers.Dropout(0.33, name='Dropout_4x4')(act_conv2d_8)

    # Downsampling: 512 feature maps
    conv2d_9 = tf.keras.layers.Conv2D(512, (4, 4), strides=(1, 1), padding='valid', name='conv_1x1')(dp_conv2d_8)
    act_conv2d_9 = tf.keras.layers.LeakyReLU(alpha=0.2, name='lReLU_1x1')(conv2d_9)
    dp_conv2d_9 = tf.keras.layers.Dropout(0.33, name='Dropout_1x1')(act_conv2d_9)

    flat_output = tf.keras.layers.Flatten()(dp_conv2d_9)
    final_output = tf.keras.layers.Dense(units=1, activation='sigmoid', name='final_output')(flat_output)

    model = tf.keras.models.Model(inputs=[input_x, input_c], outputs=final_output, name="Discriminator")
    return model


# Defining functions for loss of discriminator and generator
def discriminator_loss(real_output, fake_output):
    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Defining functions for training process:
# training_step: controls process of each training step
# train: controls training process through each epoch
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def training_step(generator_model, discriminator_model, gen_opt, disc_opt, batchsize, noise_z_dim, real_images,
                  real_labels, number_training_step):

    noise_z = tf.random.normal([batchsize, noise_z_dim])
    # Fake labels
    rnd_sample_labels = np.random.randint(0, 3, batchsize)
    # generate one_hot_encoding
    fake_labels = tf.one_hot(indices=rnd_sample_labels, depth=3, dtype=tf.float32)


    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        if number_training_step % 5 == 0:
            print("Generating fake images")
            fake_images = generator_model(inputs=[noise_z, fake_labels], training=True)
        else:
            fake_images = generator_model(inputs=[noise_z, fake_labels], training=False)
        print("Discriminating real and fake images")
        real_output = discriminator_model(inputs=[real_images, real_labels], training=True)
        fake_output = discriminator_model(inputs=[fake_images, fake_labels], training=True)
        gen_loss = generator_loss(fake_output=fake_output)
        disc_loss = discriminator_loss(real_output=real_output, fake_output=fake_output)
    tf.print(gen_loss)
    tf.print(disc_loss)

    print("Updating GEN")
    gradients_of_generator = gen_tape.gradient(gen_loss, generator_model.trainable_variables)
    gen_opt.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))

    print("Updating DISC")
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator_model.trainable_variables)
    disc_opt.apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))


def train(generator_model, discriminator_model, gen_opt, disc_opt, batchsize, noise_z_dim, real_data, epochs,
          checkpoint):
    global losses
    seed = tf.random.normal([batchsize, noise_z_dim])
    seed_ints = np.random.randint(0, 3, batchsize)
    # generate one_hot_encoding
    seed_labels = tf.one_hot(indices=seed_ints, depth=3, dtype=tf.float32)

    for epoch in range(epochs):
        print("Epoch: ", epoch)
        start = time.time()
        train_step = 1
        losses = []
        for image_batch, label_batch in real_data:
            print("Training Step: ", train_step)
            training_step(generator_model=generator_model,
                                                    discriminator_model=discriminator_model,
                                                    gen_opt=gen_opt,
                                                    disc_opt=disc_opt,
                                                    batchsize=batchsize,
                                                    noise_z_dim=noise_z_dim,
                                                    real_images=image_batch,
                                                    real_labels=label_batch,
                                                    number_training_step=train_step)
            train_step += 1


        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        generate_and_save_images(generator_model,
                                 epoch + 1,
                                 seed,
                                 seed_labels)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))


# Function to generate and save images with new test data for trained model
def generate_and_save_images(model, epoch, test_input, test_labels):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(inputs=[test_input, test_labels], training=False)

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(2, 5, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 255, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()


def main(argv=None):
    # Defining the generator and discriminator modeles
    generator = generator_model()
    discriminator = discriminator_model()


    # Hyper-parameters for training process
    EPOCHS = 5
    Batch_size = 10  # argv[2]
    noise_z_dim = 100

    # Loading dataset with helper function
    X_train, X_test, y_train, y_test = preprocess_data(class_paths=paths)

    # Take training dataset (X_train and y_train) shuffle and generated batches
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(Batch_size)

    # optimizers
    generator_optimizer = tf.keras.optimizers.Adam(0.01)
    discriminator_optimizer = tf.keras.optimizers.Adam(0.01)

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    train(generator_model=generator,
                  discriminator_model=discriminator,
                  gen_opt=generator_optimizer,
                  disc_opt=discriminator_optimizer,
                  batchsize=Batch_size,
                  noise_z_dim=noise_z_dim,
                  real_data=train_data,
                  epochs=EPOCHS,
                  checkpoint=checkpoint)

    print(losses)


if __name__ == "__main__":
    main(sys.argv)
