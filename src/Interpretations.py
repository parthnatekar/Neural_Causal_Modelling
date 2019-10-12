from sklearn.model_selection import train_test_split
from get_interventions import *
import keras
from keras import Model
import numpy as np
import tqdm
from tqdm import tqdm
import sys
from scipy.stats import pearsonr

class Interpretation_Generator():


    """
    Builds a Bayesian Causal Graph from an Artificial Neural Network model and performs causal intervention 
    to get human-understandable representations of what the model is learning.

    INPUTS

    model: A keras model instance for which causal explanations are required.
    train_data: The training dataset ([batch, height, width, channels])
    train_labels: The corresponding training labels
    bins (optional): The boundaries which will be used to bin the dataset to 
                     convert it to a discrete probability distribution.

    OUTPUTS:

    Lists containing the causal and correlational effect of a particular layer.

    """
    
    
    def __init__(self, model, train_data, train_labels, test_data = None, test_labels = None, bins = np.linspace(-1,1,num = 5)):
        self.model = model
        self.X_train = train_data
        self.y_train = train_labels
        self.X_test = test_data
        self.y_test = test_labels
        self.bins = bins
    
    def train_model(self, epochs, batch_size):
        
        if self.X_test is None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train, self.y_train, test_size=0.33, random_state=42)
        
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=epochs, batch_size=batch_size, verbose=2)
        
        self.y_pred =self.model.predict_classes(self.X_test)
        
        self.model_list = []
        
        for i in range(len(self.model.layers)):
            
            intermediate_model =  Model(inputs=self.model.input,
                                 outputs=self.model.layers[i].output)
            
            self.model_list.append(intermediate_model)
            
            intermediate_output = intermediate_model.predict(self.X_test)
            
            try:
                self.data = np.concatenate((self.data, intermediate_output), axis = 1)
            except:
                self.data = np.concatenate((self.X_test, intermediate_output), axis = 1)
            
            
        self.data = np.concatenate((self.data, self.y_pred.reshape(len(self.y_pred), 1)), axis = 1)
        #print(self.model.predict_classes(self.X_test[0].reshape((1,784))))
        
    def interpret(self, labels, layer, label):
        
        if layer == 0:
            layer_end = self.X_test.shape[1]
            layer_start = 0
            values = range(self.X_test.shape[1])
            
        else:
            layer_start = 0
            layer_start += self.X_test.shape[1]
            for i in range(len(self.model.layers[:layer-1])):
                layer_start += self.model.layers[i].output.shape[1]
            
            layer_end = layer_start + self.model.layers[layer-1].output.shape[1]
            values = range(self.model.layers[layer-1].output.shape[1])
            
        print(layer_start, layer_end, values)
        
        
        y_pred_prob = self.model.predict(self.X_test)[:, label]
        
        self.correlational_effect = []

        with tqdm(total=len(values), file=sys.stdout) as pbar:
            for i in values:
                #print(pearsonr(data[:, i], data[:, -1]))
                self.correlational_effect.append(pearsonr(self.data[:,layer_start+i], y_pred_prob)[0])
                pbar.set_description('processed: %d' % (1 + i))
                pbar.update(1)
                
        
        dict_list = get_dict_list(self.data, labels, self.bins)
            
        layer_dict = create_histogram_dict(self.data[:, 0:layer_end], self.bins)

        self.causal_effect = []

        with tqdm(total=len(values), file=sys.stdout) as pbar:
            for i in values:
                (_, average_causal_effect) = return_causal_effect(self.data, dict_list, layer_dict, i,  [layer_start, layer_end], self.bins, labels)
                self.causal_effect.append(average_causal_effect[-1,label])
                pbar.set_description('processed: %d' % (1 + i))
                pbar.update(1)
                
        
        return(self.causal_effect, self.correlational_effect)
    
    
    def perturbation_tester(self, SubModel,label, layer = 0):
        
        if layer == 0:
            array = self.X_test
        else:
            array = self.model_list[layer-1].predict(self.X_test.reshape((-1,784)))
    
        pixels_og = np.array(np.nan_to_num(self.correlational_effect))
        
        pixels = np.array(np.nan_to_num(self.causal_effect))
        #pixels_og = pixels_og/np.max(pixels_og)
        #pixels = pixels/np.max(pixels)
        print('Original Predictions:', (SubModel.predict(pixels_og.reshape((1,array.shape[1])))[0][label]), (SubModel.predict(pixels.reshape((1,array.shape[1])))[0][label]))

        indices = []

        original_average = 0
        
        for i in range(len(self.X_test)):
            if self.model.predict_classes(self.X_test[i].reshape((1,784))) == label:
                
                indices.append(i)
                original_average +=self.model.predict(self.X_test[i].reshape((1,784)))[0][label]

        original_average = original_average/len(indices)
        #print(original_average)
        average1= 0
        average2 = 0

        max_indices = np.argsort(pixels)[-40:]
        print(max_indices)

        for item in indices:

            instance = array[item].copy()

            correlation_average = 0
            causal_average = 0

            count1 = 0
            count2 = 0

            for i in range(len(instance)):

                if pixels_og[i] > 0 and i not in max_indices:

                    instance[i] = 0
                    count1 +=1
                    
            average1 += (SubModel.predict(instance.reshape((1,array.shape[1])))[0][label])

            instance = array[item].copy()

            for index in max_indices:

                    instance[index] = 0

                    count2 +=1
            average2 += (SubModel.predict(instance.reshape((1,array.shape[1])))[0][label])

        self.causal_perturbation = average1/len(indices)
        self.correlational_perturbation = average2/len(indices)
        
        print(count1,count2)
        
        return(self.causal_perturbation, self.correlational_perturbation)
