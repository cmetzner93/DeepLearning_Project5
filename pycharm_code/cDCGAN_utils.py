import matplotlib.pyplot as plt


# Function to generate and save images with new test data for trained model
def generate_and_save_images(model, epoch, test_input, test_labels):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(inputs=[test_input, test_labels], training=False)

    fig = plt.figure(figsize=(4, 5))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 5, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 255, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()