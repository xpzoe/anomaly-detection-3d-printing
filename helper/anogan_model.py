import tensorflow as tf
from tensorflow import keras
from keras import layers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Generator(keras.Model):

    def __init__(self, gf_dim):
        super(Generator, self).__init__()

        self.fc = layers.Dense(4*4*512)

        self.conv1 = layers.Conv2DTranspose(gf_dim*8, 3, 1, 'valid')  # o = (i-1)*s+2*p+k
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2DTranspose(gf_dim*4, 4, 2, 'valid')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2DTranspose(gf_dim*2, 4, 2, 'valid')
        self.bn3 = layers.BatchNormalization()
        self.conv4 = layers.Conv2DTranspose(gf_dim, 4, 2, 'valid')
        self.bn4 = layers.BatchNormalization()
        self.conv5 = layers.Conv2DTranspose(1, 6, 2, 'valid')

    def call(self, inputs, training=None):
        x = self.fc(inputs)
        x = tf.reshape(x, [-1, 4, 4, 512])
        x = tf.nn.leaky_relu(x)
        x = tf.nn.leaky_relu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        x = tf.nn.leaky_relu(self.bn4(self.conv4(x), training=training))
        x = self.conv5(x)
        x = tf.tanh(x)
        return x


class Discriminator(keras.Model):
    def __init__(self, df_dim):
        super(Discriminator, self).__init__()

        # [b,128,128,1] => [b,1]
        self.conv1 = layers.Conv2D(df_dim, 4, 2, 'valid')  # o = (i-k+s)/s

        self.conv2 = layers.Conv2D(df_dim*2, 4, 2, 'valid')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(df_dim*4, 4, 2, 'valid')
        self.bn3 = layers.BatchNormalization()

        self.conv4 = layers.Conv2D(df_dim*8, 4, 2, 'valid')
        self.bn4 = layers.BatchNormalization()

        self.conv5 = layers.Conv2D(df_dim * 16, 4, 1, 'valid')
        self.bn5 = layers.BatchNormalization()

        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)

    def call(self, inputs, training=None):
        x = tf.nn.dropout(tf.nn.leaky_relu(self.conv1(inputs)), 0.25)
        x = tf.nn.dropout(tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training)), 0.25)
        x = tf.nn.dropout(tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training)), 0.25)
        x = tf.nn.dropout(tf.nn.leaky_relu(self.bn4(self.conv4(x), training=training)), 0.25)
        x = tf.nn.dropout(tf.nn.leaky_relu(self.bn5(self.conv5(x), training=training)), 0.25)
        x = self.flatten(x)
        logits = self.fc(x)
        return logits


def print_net():
    g = Generator(64)
    g.build(input_shape=[None, 100])
    d = Discriminator(64)
    d.build(input_shape=[None, 64, 64, 1])
    g.summary()
    d.summary()


