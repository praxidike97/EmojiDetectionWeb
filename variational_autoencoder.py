import numpy as np
import argparse

import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Input, Conv2D, Flatten, Conv2DTranspose, Reshape
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.metrics import binary_crossentropy
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import Callback

import matplotlib.pyplot as plt


batch_size = 32
latent_dim = 2
img_shape = np.array([28, 28])

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1] * X_train.shape[2])) / 255.
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2])) / 255.
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))/255.
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))/255.

class PlotCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch%10 == 0:
            prediction = self.model.predict(X_test[:batch_size])
            print(prediction.shape)

            row_size = 3
            ig, axs = plt.subplots(row_size, row_size)
            print(y_test[:10])
            for x in range(row_size):
                for y in range(row_size):
                    #axs.subplot(row_size**2, row_size**2, row_size*x+y+1)
                    axs[x, y].imshow(np.reshape(prediction[x*row_size+y], (28, 28)), cmap='gray', vmin=0., vmax=1.)
            plt.show()


def sample(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim))
    return z_mean + K.exp(z_log_sigma) * epsilon


def vae_loss(z_mean, z_log_sigma):
    def loss(input, output):
        input = K.flatten(input)
        output = K.flatten(output)

        xent_loss = binary_crossentropy(input, output)*np.prod(img_shape)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        print("kl_loss: {}".format(xent_loss + kl_loss))
        return xent_loss + kl_loss
    return loss


def create_vae(input_dim):
    print((batch_size, ) + tuple(img_shape.tolist()) + (1, ))
    input_layer = Input(batch_shape=(batch_size, ) + tuple(img_shape.tolist()) + (1, ))
    conv01 = Conv2D(64, (2, 2), activation='relu', padding='same')(input_layer)
    conv02 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv01)
    conv03 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv02)

    flatten01 = Flatten()(conv03)

    dense01 = Dense(units=32, activation='relu')(flatten01)

    z_mean = Dense(units=latent_dim)(dense01)
    z_log_sigma = Dense(units=latent_dim)(dense01)

    lambda_layer = Lambda(sample, output_shape=(latent_dim, ))([z_mean, z_log_sigma])

    dense02 = Dense(units=32, activation='relu')
    dense03 = Dense(units=np.prod(img_shape), activation='relu')
    reshape = Reshape(tuple(img_shape.tolist()) + (1, ))
    conv04 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')
    conv05 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')
    conv06 = Conv2DTranspose(64, (2, 2), activation='relu', padding='same')
    output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')

    x = dense02(lambda_layer)
    x = dense03(x)
    x = reshape(x)
    x = conv04(x)
    x = conv05(x)
    x = conv06(x)
    output_x = output_layer(x)

    vae = Model(input_layer, output_x)

    input_generator = Input(batch_shape=(batch_size, latent_dim))
    x = dense02(input_generator)
    x = dense03(x)
    x = reshape(x)
    x = conv04(x)
    x = conv05(x)
    x = conv06(x)
    output_x = output_layer(x)

    generator = Model(input_generator, output_x)

    print(vae.summary())
    vae.compile(optimizer=RMSprop(), loss=vae_loss(z_mean, z_log_sigma))
    generator.compile(optimizer=RMSprop(),loss='mse')

    return vae, generator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    args.generate = True

    if args.train:
        vae, generator = create_vae(input_dim=X_train.shape[-1])
        vae.fit(x=X_train, y=X_train, batch_size=batch_size, shuffle=True,
                epochs=20, callbacks=[PlotCallback()])
        generator.save('generator.h5')
    elif args.generate:
        generator = load_model('generator.h5')
        row_size = 15

        v_min = -1
        v_max = 1
        x = np.linspace(-1, 1, row_size)
        x = np.reshape(x, (len(x), 1))
        y = np.linspace(-1, 1, row_size)
        y = np.reshape(x, (len(y), 1))
        xx, yy = np.meshgrid(x, y)

        ig, axs = plt.subplots(row_size, row_size)
        print(y_test[:10])
        for row in range(row_size):
            for column in range(row_size):

                latent_vector = np.array([xx[row,column], yy[row,column]])
                print(latent_vector.shape)
                print(latent_vector)
                latent_vector = np.repeat(latent_vector[:,np.newaxis], batch_size, axis=1).T
                print(latent_vector.shape)
                print(latent_vector)

                prediction = generator.predict(latent_vector)
                axs[row, column].imshow(np.squeeze(prediction[0]), cmap='gray', vmin=0., vmax=1.)
        plt.savefig("generated_img.png")
