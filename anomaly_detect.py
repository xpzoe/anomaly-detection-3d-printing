import numpy as np
import tensorflow as tf
from anogan_model import Generator, Discriminator
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
from dataload import preprocess_my_image
from PIL import Image


def load_model():
    d = Discriminator(64)
    d.build(input_shape=(None, 128, 128, 1))
    g = Generator(64)
    g.build(input_shape=(None, 100))
    g_optimizer = tf.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
    d_optimizer = tf.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    g.compile(loss='binary_crossentropy', optimizer=g_optimizer)
    d.compile(loss='binary_crossentropy', optimizer=d_optimizer)
    d.load_weights('./weights20220210-015855/discriminator.h5')
    g.load_weights('./weights20220210-015855/generator.h5')
    return g, d


def extract_feature(discriminator):
    dInput = keras.Input(shape=(128, 128, 1,))
    dInput = tf.nn.sigmoid(dInput)
    x = tf.nn.dropout(tf.nn.leaky_relu(discriminator.conv1(dInput)), 0.25)
    x = tf.nn.dropout(tf.nn.leaky_relu(discriminator.bn2(d.conv2(x), training=False)), 0.25)
    x = discriminator.conv3(x)
    intermidiate_model = keras.Model(inputs=dInput, outputs=x)
    intermidiate_model.compile(loss='binary_crossentropy', optimizer='Adam')
    intermidiate_model.trainable = False
    return intermidiate_model


def detect_model(intermidiate_model, generator):
    gInput = keras.Input(shape=(10,))
    gInput = keras.layers.Dense(100, trainable=True)(gInput)
    gInput = tf.nn.sigmoid(gInput)
    gOut = generator(gInput)
    fOut = intermidiate_model(gOut)
    model = keras.Model(inputs=gInput, outputs=[gOut, fOut])
    return model


def sum_of_residual(y_true, y_pred):
    res = tf.reduce_sum(tf.abs(y_true - y_pred))
    return res

def main():

    g, d = load_model()
    g.trainable, d.trainable = False, False
    f = extract_feature(d)
    model = detect_model(f, g)  # gout:kerastensor, fout:kerastensor
    model.trainable = True
    model_optimizer = tf.optimizers.Adam(learning_rate=0.00005, beta_1=0.5)
    epochs = 100

    images = preprocess_my_image()
    print(tf.shape(images)[0].numpy())
    mean = tf.reshape(tf.reduce_mean(images, 0), [1, 128, 128, 1])

    random_test_pos = np.random.randint(tf.shape(images)[0].numpy(), size=10)
    marks = []
    similars = []
    for i, pos in enumerate(random_test_pos):
        test_img = tf.reshape(images[pos], [1, 128, 128, 1])
        weighted_mean = 0.7 * test_img + 0.3 * mean
        print('start test:', i)

        for epoch in range(epochs):

            z = tf.random.uniform([1, 100], minval=-1., maxval=1.)
            f_x = f(weighted_mean)

            with tf.GradientTape() as tape:
                similar_img, f_z = model(z)
                loss_r = sum_of_residual(tf.cast(weighted_mean, dtype='float32'), similar_img)
                loss_d = sum_of_residual(f_x, f_z)
                loss = 0.6 * loss_r + 0.4 * loss_d
            grads = tape.gradient(loss, model.trainable_variables)
            model_optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if (epoch + 1) % 10 == 0:
                print('test:', i, 'epoch', epoch, 'loss:', float(loss))

        np_residual = tf.squeeze(tf.abs(tf.cast(test_img, dtype='float32') - similar_img)).numpy() * 65535
        #  np_residual = tf.abs(test_img - weighted_mean)
        original_x = (test_img.numpy() * 32767.5 + 32767.5)
        similar_x = (similar_img.numpy() * 32767.5 + 32767.5)

        pos = np.where(np_residual > 21000, 1, 0)
        #img = Image.fromarray(np.uint8(tf.squeeze(original_x)))
        mark = Image.new('RGB', (128, 128), color=0)
        mark_array = np.array(mark)
        for x in range(127):
            for y in range(127):
                if pos[x, y] == 1:  mark_array[x, y] = [255, 0, 0]
        marks.append(np.array(mark_array))
        similars.append(tf.squeeze(similar_x))

    plt.figure(1)
    for i in range(10):
        plt.subplot(5, 4, i * 2 + 1)
        plt.title('query image')
        plt.imshow(tf.squeeze(images[random_test_pos[i]]), cmap=plt.cm.gray)

        plt.subplot(5, 4, i * 2 + 2)
        plt.imshow(marks[i], cmap=plt.cm.gray)

    plt.figure(2)
    for i in range(10):
        plt.subplot(5, 2, i + 1)
        plt.title('similar image')
        plt.imshow(tf.squeeze(similars[i]), cmap=plt.cm.gray)
    plt.show()

if __name__ == '__main__':
    main()
