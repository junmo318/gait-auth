import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

BUFFER_SIZE = 60000
BATCH_SIZE = 256

def main():
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)



    generator = make_generator_model()

    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)

    plt.imshow(generated_image[0, :, :, 0], cmap='gray')



    discriminator = make_discriminator_model()
    decision = discriminator(generated_image)
    print (decision)



    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 16

    # We will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".


    train(train_dataset, EPOCHS)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


if __name__ == "__main__":
    print("No test sequence.")
