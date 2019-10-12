import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D, UpSampling2D
from keras.utils import np_utils
from keras.models import Model
from keras import regularizers, losses
from keras.regularizers import Regularizer
from keras import backend as K
from causal_inferences import calculate_causal
from get_interventions import *
from tqdm import tqdm
import os
from triplet_loss_moindrot import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
os.environ["CUDA_VISIBLE_DEVICES"]="0, 3"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)
from sklearn.preprocessing import MinMaxScaler
import pickle
import gzip
import sys
K.set_image_dim_ordering('th')


class Automatic_Autoencoder():

"""
A class for automatically generating an auxiliary autoencoded distribution from a pretrained keras model instance

INPUTS:
    model: A keras model instance
    X_train: Input training data ([batch, height, width, channels])
    y_train: Input training labels
    baseline_distribution: Initial output distribution predicted by the original model for taking KL-divergence loss
    baseline_weights: Path of saved weights of original model

OUTPUTS:
    New autoecoded distribution saved as a numpy array
"""


    def __init__(self, model, X_train, X_test, y_train, baseline_distribution, baseline_weights):
        self.baseline_model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.baseline_distribution = baseline_distribution
        self.baseline_weights = baseline_weights

    def custom_loss(embeddings, m = 1.0):
        #embeddings = tf.nn.l2_normalize(embeddings, axis = 1)
        
        def faithfulness_loss(y_true, y_pred):
            
            
            loss, fraction_positive = batch_all_triplet_loss(y_true,embeddings,margin=m)
            #loss = K.expand_dims(loss, axis = -1)
            #print(loss.get_shape())
            return(loss)
            
        return(faithfulness_loss)

    def build_auxiliary_model(self)

        for j in range(1, len(self.baseline_model.layers)):
            
            if len(list(self.baseline_model.layers[j].output.shape)) == 4:
            
                temp_model = keras.models.clone_model(self.baseline_model)

                x = temp_model.input

                for i in range(1, len(temp_model.layers)):

                    if x.name.split('_')[0] == ('conv%d' %j):

                        intermediate_model = Model(inputs=self.baseline_model.input,
                                                     outputs=self.baseline_model.get_layer('conv%d' %j).output)
                        cae_output = intermediate_model.predict(X_train)

                        autoencoder_0 = Conv2D(8, kernel_size=5, activation='tanh', name = 'cae_0', padding = 'same', activity_regularizer = keras.regularizers.l1(10e-8))(x)
                        autoencoder_1 = Conv2D(1, kernel_size=5, activation='tanh', name = 'embedding', padding = 'same', activity_regularizer = keras.regularizers.l1(10e-8))(autoencoder_0)
                        autoencoder_2 = Conv2D(8, kernel_size=5, activation='tanh', name = 'cae_2', padding = 'same')(autoencoder_1)
                        autoencoder_3 = Conv2D(int(self.baseline_model.get_layer('conv%d' %j).output.shape[1]), kernel_size=5, activation='tanh', name = 'reconstruction', padding = 'same')(autoencoder_2)

                        x = temp_model.layers[i](autoencoder_3)

                    else:

                        x = temp_model.layers[i](x)

                model = Model(inputs=temp_model.input, outputs=[x, autoencoder_1, autoencoder_3])
                model.load_weights('baseline.h5', by_name=True)

                print(model.summary())

                loss = custom_loss(tf.nn.l2_normalize(K.reshape(autoencoder_1, (K.shape(autoencoder_1)[0],784)), axis = 1))

                model.compile(optimizer='adam', loss={'output': 'kullback_leibler_divergence', 'embedding': loss, 
                                                  'reconstruction': 'mean_absolute_error'}, loss_weights=[0.5, 1., 0.05])

                model.fit(self.X_train,
                      {'output': self.baseline_distribution, 'embedding': self.y_train,
                       'reconstruction': cae_output},
                      epochs=30, batch_size=200, verbose = 2)
                output = model.predict(self.X_test)

                try:
                    data = np.concatenate((data, output[1].reshape((-1, output[1].shape[2]*output[1].shape[3]))), axis = 1)
                    print(model.layers[i].name, data.shape)
                except Exception as e:
                    data = np.concatenate((self.X_test.reshape((-1, X_test.shape[2]*X_test.shape[3])), output[1].reshape((-1, output[1].shape[2]*output[1].shape[3]))), axis = 1)
                    print(e, data.shape)


        y_pred = np.argmax(output[0], axis = 1)
        data = np.concatenate((data, y_pred.reshape(len(y_pred), 1)), axis = 1)

        np.save('autoencoded_distribution.npy', data)

if __name__ == '__main__':

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # load data
    from keras.datasets import fashion_mnist
    ((X_train, y_train), (X_test, y_test)) = fashion_mnist.load_data()

    # flatten 28*28 images to a 784 vector for each image
    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

    # normalize inputs from 0-255 to 0-1
    scaler = MinMaxScaler(feature_range = (-1,1))

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # reshape to be [samples][pixels][width][height]
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
    y_train_int = y_train

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    #define main flow
    main_input = Input(shape=(1,28,28))
    conv1 = Conv2D(32, kernel_size=5, activation='tanh', name = 'conv1', padding = 'same')(main_input)
    conv2 = Conv2D(16, kernel_size=5, activation='tanh', name = 'conv2', padding = 'same')(conv1)
    conv3 = Conv2D(16, kernel_size=5, activation='tanh', name = 'conv3', padding = 'same')(conv2)
    conv4 = Conv2D(8, kernel_size=5, activation='tanh', name = 'conv4', padding = 'same')(conv3)
    conv5 = Conv2D(1, kernel_size=5, activation='tanh', name = 'conv5', padding = 'same')(conv4)
    x = Flatten(name = 'flatten')(conv5)
    x = Dense(128, activation='tanh', name = 'dense')(x)
    main_output = Dense(num_classes, activation='softmax', name = 'output')(x)

    #define baseline model
    baseline_model = Model(inputs = main_input, output = main_output)

    baseline_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    baseline_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=200, verbose=2)

    baseline_distribution = baseline_model.predict(X_train)

    baseline_model.save('baseline.h5')

    E = Automatic_Autoencoder(baseline_model, X_train, X_test, y_train, baseline_distribution, baseline_weights)

    E.build_auxiliary_model()