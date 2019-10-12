import sys
import numpy
import os
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras import backend as K
import gzip
import pickle
from get_interventions import *
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler, scale
from tqdm import tqdm
from scipy.stats import pearsonr
from interpretations import *

# fix random seed for reproducibility
seed = 8
numpy.random.seed(seed)
# load data
os.environ["CUDA_VISIBLE_DEVICES"]="4"
​
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, test_set = pickle.load(f, encoding='latin1')
    
X_train,y_train = train_set
​
X_test,y_test = test_set
​
# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
# normalize inputs from 0-255 to 0-1
​
scaler = MinMaxScaler(feature_range = (-1,1))
​
#X_train = X_train / 255
#X_test = X_test / 255
​
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
​
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
​
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels//4, input_dim=num_pixels, kernel_initializer='normal', activation='tanh', name = 'layer1'))
    model.add(Dense(num_pixels//4, input_dim=num_pixels, kernel_initializer='normal', activation='tanh', name = 'layer2'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax', name = 'layer3'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
​
# build the model
model = baseline_model()
​
generator = Interpretation_Generator(model, X_train, y_train, X_test, y_test)
​
generator.train_model(10, 200)
​
generator.model.save('model_3.h5')
​
​
def sub_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels//4, input_dim=num_pixels, kernel_initializer='normal', activation='tanh', name = 'layer1'))
    model.add(Dense(num_pixels//4, input_dim=num_pixels//4, kernel_initializer='normal', activation='tanh', name = 'layer2'))
    model.add(Dense(num_classes, input_dim=num_pixels//4, kernel_initializer='normal', activation='softmax', name = 'layer3'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
​
SubModel = sub_model()
SubModel.load_weights('model_3.h5', by_name=True)
​
compare = []
images = []
for i in range(10):
    causal_effect, correlational_effect = generator.interpret([0,1,2,3,4,5,6,7,8,9], layer = 0, label = i)
    causal_perturbation, correlational_perturbation = generator.perturbation_tester(SubModel, i, layer = 0)
    compare.append((causal_perturbation, correlational_perturbation))
    images.append((causal_effect, correlational_effect))
print(compare)
