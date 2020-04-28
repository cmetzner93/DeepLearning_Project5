import tensorflow as tf
import numpy as np
import time
from cDCGAN_models import generator_model, discriminator_model, gan_model
from cDCGAN_data_prep import preprocess_data
from cDCGAN_utils import generate_and_save_images, save_diagnostics_to_file

# Load Mammography (images) dataset including their respective labels (one-hot-encoded)
# Images are represented as greyscale using RGB-values
# Make sure directories are correct: Directory paths can be changed in script "cDCGAN_data_prep.py"
X_train, X_test, y_train, y_test = preprocess_data()

# Set Hyper-parameters for training the cDCGAN
buffer_size = len(X_train) + 1  # Shuffle training data, adding 1 enables uniform shuffle
print(len(X_train))             # (every random permutation is equally likely to occur)
batch_size = 20                 # Split training set (real images and respective labels) into batches
EPOCHS = 50                     # Number of epochs of training
dim_noise_z = 100               # Size of latent space (noise z) used to map fake mammography images

# Use tf.data.Dataset.from_tensor_slices to shuffle data (uniformly) and create an tensor object which holds all
# batches containing the image data and their respective labels for given batch_size
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=buffer_size).batch(
    batch_size=batch_size)
"""
# The following for-loop print the data (RGB-values) for each batch and their respective labels.
for image_batch, label_batch in train_data.take(5):
    #print(image_batch)
    tf.print(label_batch)
"""

# Define cDCGAN composites
generator = generator_model()
# print(generator.summary())
discriminator = discriminator_model()
# print(discriminator.summary())
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['acc'])
gan = gan_model(g_model=generator, d_model=discriminator)

# Code to keep track of training process
# Stores losses and accuracy
diagnostic_info = []

# Training the conditional DCGAN
# Loop through number of epochs
for epoch in range(EPOCHS):
    print("Current epoch: {}".format(epoch+1))
    start = time.time()
    train_step = 1

    # Lists for storing generator and discriminator losses
    gen_losses_per_epoch = []
    disc_losses_per_epoch = []

    seed = tf.random.normal([batch_size, dim_noise_z])
    seed_ints = np.random.randint(0, 3, batch_size)
    # generate one_hot_encoding
    seed_labels = tf.one_hot(indices=seed_ints, depth=3, dtype=tf.float32)

    diagnostics_per_epoch = []

    for image_batch, label_batch in train_data:
        print('Current training step: ', train_step)

        # Generate tensor holding specific number of (batch_size) latent vectors of certain dimension (dim_noise_z) for
        # fake image generation
        noise_z = tf.random.normal([batch_size, dim_noise_z])
        # Generate randomly integers to be used for one-hot-encoding of fake labels as input for generator
        # Classes are as following: Normal --> 0 --> [1,0,0]; Benign --> 1 --> [0,1,0]; Malignant --> 2 --> [0,0,1]
        fake_labels_as_int = np.random.randint(low=0, high=3, size=batch_size)
        fake_labels = tf.one_hot(indices=fake_labels_as_int, depth=3, dtype=tf.float32)

        # Generating a set of fake images
        print("Generate fake images")
        fake_images = generator.predict([noise_z, fake_labels], verbose=0)

        # generate labels to mark real images as real
        print("Real and Fake loss")
        y_real = [1]*batch_size
        disc_real_loss = discriminator.train_on_batch([image_batch, label_batch], y_real)
        y_fake = [0]*batch_size
        disc_fake_loss = discriminator.train_on_batch([fake_images, fake_labels], y_fake)

        print("Generator loss")
        gan_loss = gan.train_on_batch([noise_z, fake_labels], y_real)

        # Storing batch metrics in list
        time_stamp = time.time() - start
        diagnostics_per_batch = [disc_real_loss, disc_fake_loss, gan_loss, time_stamp]
        # appending batch metrics to epoch metrics
        diagnostics_per_epoch.append(diagnostics_per_batch)

        train_step += 1

        # Call function to generate random images with noise z and fake labels to check training evolution of the cDCGAN
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed,
                                 seed_labels)


    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
    print()
    diagnostic_info.append(diagnostics_per_epoch)

# Calls function to write diagnostic information in text file in working directory
save_diagnostics_to_file('cDCGAN_diagnostics', diagnostic_info)

# Saving models for reproduction in working directory
generator.save('cDCGAN_generator', save_format='h5')
discriminator.save('cDCGAN_discriminator', save_format='h5')
gan.save('cDCGAN_gan', save_format='h5')
