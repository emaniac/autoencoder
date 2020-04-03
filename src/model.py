import tensorflow as tf

import matplotlib.pyplot as plt

def show_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model accuracy and loss')
    plt.ylabel('Accuracy, Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train loss', 'Val loss'], loc='upper left')
    plt.show()


def Encoder(N):

    model = tf.keras.models.Sequential(name='encoder')

    model.add(tf.keras.layers.Conv2D(input_shape=(32, 32, 3),
                                     filters=32, 
                                     kernel_size=(3, 3),
                                     activation=tf.nn.relu,
                                     name='encoder_conv1'))

    model.add(tf.keras.layers.Conv2D(filters=16, 
                                     kernel_size=(3, 3),
                                     activation=tf.nn.relu,
                                     name='encoder_conv2'))

    model.add(tf.keras.layers.Conv2D(filters=8, 
                                     kernel_size=(3, 3),
                                     activation=tf.nn.relu,
                                     name='encoder_conv3'))

    model.add(tf.keras.layers.Flatten(name='encoder_flatten'))

    model.add(tf.keras.layers.Dense(units=N,
                                    activation=tf.nn.sigmoid,
                                    name='encoder_dense'))

    return model


def Decoder(N):

    model = tf.keras.models.Sequential(name='decoder')

    model.add(tf.keras.layers.Dense(input_shape=(N,),
                                    units=5408,
                                    activation=tf.nn.sigmoid,
                                    name='decoder_dense1'))

    model.add(tf.keras.layers.Reshape((26, 26, 8), name='decoder_reshape1'))

    model.add(tf.keras.layers.Conv2DTranspose(filters=16, 
                                              kernel_size=(3, 3), 
                                              activation=tf.nn.relu,
                                              name='decoder_conv1'))

    model.add(tf.keras.layers.Conv2DTranspose(filters=32, 
                                              kernel_size=(3, 3), 
                                              activation=tf.nn.relu,
                                              name='decoder_conv2'))

    model.add(tf.keras.layers.Conv2DTranspose(filters=3, 
                                              kernel_size=(3, 3), 
                                              activation=tf.nn.sigmoid,
                                              name='decoder_conv3'))

    return model


def Autoencoder(N):

    model = tf.keras.models.Sequential(name='autoencoder')
    model.add(Encoder(N))
    model.add(Decoder(N))
    return model

