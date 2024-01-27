import os
import numpy as np
import tensorflow as tf
from anogan_model import Generator, Discriminator
from dataload import preprocess_my_image
import datetime
from PIL import Image


def celoss_ones(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)


def d_loss_fn(generator, discraminator, batch_z, batch_x, is_training):
    # 1.treat real image as real
    # 2.treat generated image as fake
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discraminator(fake_image, is_training)
    d_real_logits = discraminator(batch_x, is_training)

    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)

    loss = d_loss_real + d_loss_fake
    return loss


def g_loss_fn(generator, discraminator, batch_z, is_training):
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discraminator(fake_image, is_training)
    loss = celoss_ones(d_fake_logits)
    return loss


def main():
    log_dir = 'tfBoardlog/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(log_dir)

    z_dim = 100
    epochs = 3500
    batch_size = 10
    learning_rate_g = 0.0001
    learning_rate_d = 0.0001
    is_training = True
    gf_dim = 64
    df_dim = 64

    x = preprocess_my_image()
    print("dataset shape is:", x.shape)
    train_db = tf.data.Dataset.from_tensor_slices(x)
    train_db = train_db.batch(batch_size)

    generator = Generator(gf_dim)
    generator.build(input_shape=(None, z_dim))
    discriminator = Discriminator(df_dim)
    discriminator.build(input_shape=(None, 128, 128, 1))

    g_optimizer = tf.optimizers.Adam(learning_rate=learning_rate_g, beta_1=0.5)
    d_optimizer = tf.optimizers.Adam(learning_rate=learning_rate_d, beta_1=0.5)

    for epoch in range(epochs):
        for step, batch_x in enumerate(train_db):
            batch_z = tf.random.uniform([batch_size, z_dim], minval=-1., maxval=1.)

            #  train D
            with tf.GradientTape() as tape:
                d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

            #  train G
            with tf.GradientTape() as tape:
                g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
            grads = tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('d-loss', float(d_loss), step=epoch)
            tf.summary.scalar('g-loss', float(g_loss), step=epoch)

        if (epoch + 1) % 10 == 0:
            print(epoch, 'd-loss:', float(d_loss), 'g-loss:', float(g_loss))

        if (epoch + 1) % 100 == 0:
            z = tf.random.uniform([1, z_dim], minval=-1., maxval=1.)
            rec_image = generator(z,training=False)*65535.
            rec_image = tf.cast(rec_image, np.uint8)
            Image.fromarray(tf.squeeze(rec_image).numpy()).save('rec_images_0902/rec_epoch_%d.tif' % epoch)

    weights_dir = 'weights' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.mkdir(weights_dir)
    generator.save_weights(weights_dir + '/' + 'generator.h5', True)
    discriminator.save_weights(weights_dir + '/' + 'discriminator.h5', True)


if __name__ == '__main__':
    main()
