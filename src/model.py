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


class Encoder(tf.keras.layers.Layer):

    def __init__(self, intermediate_dim):
        super(Encoder, self).__init__()

        self.hidden_layer = tf.keras.layers.Dense(
            units=intermediate_dim,
            activation=tf.nn.relu,
            kernel_initializer='he_uniform')

        self.conv_layer = tf.keras.layers.Conv1D(filters=32, 
                                          kernel_size=(3,), 
                                          activation=tf.nn.sigmoid)

        self.output_layer = tf.keras.layers.Dense(
            units=intermediate_dim,
            activation=tf.nn.sigmoid)
        
    def call(self, input_features):
        activation = self.conv_layer(input_features)
        activation = self.hidden_layer(activation)
        return self.output_layer(activation)

class Decoder(tf.keras.layers.Layer):
    
    def __init__(self, intermediate_dim, original_dim):
        super(Decoder, self).__init__()


        self.hidden_layer = tf.keras.layers.Dense(
            units=intermediate_dim,
            activation=tf.nn.relu,
            kernel_initializer='he_uniform')

        self.output_layer = tf.keras.layers.Dense(
            units=original_dim,
            activation=tf.nn.sigmoid)
    
    def call(self, code):
        activation = self.hidden_layer(code)
        return self.output_layer(activation)


class Autoencoder(tf.keras.Model):

    def __init__(self, intermediate_dim, original_dim):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(intermediate_dim=intermediate_dim)
        self.decoder = Decoder(intermediate_dim=intermediate_dim, original_dim=original_dim)
  
    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed
