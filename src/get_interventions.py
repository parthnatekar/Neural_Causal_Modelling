import numpy as np
import time

def main():
    pass

np.random.seed(7)

def create_histogram(array, bins):
    # generates an n-dimensional sparse histogram from an array having n columns
    # Only call for low number of features (<10). For any number of features higher than this
    # use the function "create_histogram_dict()"
    number_of_bins = len(bins) - 1
    bin_indices = {}
    hist = np.zeros((number_of_bins,)*len(array[0]))

    for x in array:
        binned = tuple(np.digitize(x, bins)-1)
        if binned in bin_indices.keys():
            bin_indices[binned] = bin_indices[binned] + 1
        else:
            bin_indices[binned] = 1

    for k,v in bin_indices.items():
        np.put(hist, np.ravel_multi_index(k, hist.shape), v)

    return(hist, bin_indices)

def create_histogram_dict(array, bins):
    #creates a dictionary containing indexes of non-zero entries of sparse n-dimensional histogram as keys and their frequency as values.
    #Intended to replace an n-dimensional histogram. Much faster since there is no need to create an n-dimensional array object
    number_of_bins = len(bins)
    bin_indices = {}
    
    for x in array:
        binned = tuple(np.digitize(x, bins)-1)
        if binned in bin_indices.keys():
            bin_indices[binned] = bin_indices[binned] + 1
        else:
            bin_indices[binned] = 1

    return(bin_indices)

def get_dict_list(data, labels, bins):
    
    data_list = []
    dict_list = []
    
    for label in labels:
        data_list.append(data[data[:, -1] == label])
    
    for item in data_list:
        item = item[:, :-1]
        dict_list.append(create_histogram_dict(item, bins))

    return(dict_list)


def return_causal_effect(data, dict_list, layer_dict, node, layer_indices, bins, labels):
    #Given a list of histogram dictionaries and the node/feature, calculates causal effect of feature on output labels
    
    layer = data[:, 0:layer_indices[1]]
    number_of_bins = len(bins)-1
    #dict_list = get_dict_list(data, labels, bins)
    #layer_dict = create_histogram_dict(layer, bins)
    
    sufficient_set = np.delete(layer, layer_indices[0]+node, axis = 1)
    sufficient_set_dict = create_histogram_dict(sufficient_set, bins)
    #sufficient_set_list = get_dict_list(np.concatenate((sufficient_set, data[:, -1].reshape(len(data[:, -1]), 1)), axis = 1), labels, bins)
    #print(layer_dict, sufficient_set_dict)
    layer_sum = sum(list(layer_dict.values()))
    #print("layer sum = ", layer_sum)
    #print((sufficient_set_list))

    conditional_probability = np.zeros((number_of_bins, len(dict_list)))
    causal_effect = np.zeros((number_of_bins, len(dict_list)))
    conditional_sum = np.zeros((number_of_bins, len(dict_list)))
    non_conditional_sum = np.zeros((len(dict_list)))
    non_conditional_probability = np.zeros((len(dict_list)))
    prior_sum = np.zeros((number_of_bins, len(dict_list)))
    prior_probability = np.zeros((number_of_bins, len(dict_list)))

    for j in range(len(dict_list)):
        for key, val in dict_list[j].items():
            non_conditional_sum[j] = non_conditional_sum[j] + val

    for j in range(len(dict_list)):
        non_conditional_probability[j] = (non_conditional_sum[j]) / (np.sum(non_conditional_sum) + 10e-7)
    
    #print(non_conditional_probability, non_conditional_sum)
    
    
    for i in range(number_of_bins):

        for j in range(len(dict_list)):
            
            term1 = 0
            term2 = 0
            count = 0
            for key,val in dict_list[j].items():
                if key[layer_indices[0]+node] == i: 
                    count +=1
                    term1 = term1 + val * sufficient_set_dict[key[0:node+layer_indices[0]]+key[layer_indices[0]+node+1:layer_indices[1]]] / layer_dict[tuple(x for x in key[0:layer_indices[1]])] / len(data[:, 0])
                    
                    #term2 = term2 + 1 / len(data[:, 0])
                    
                    #print(term1, term2)
            causal_effect[i][j] = term1 #- term2
            #print(count, term1)
    causal_means = np.mean(causal_effect, axis = 0)
            
    #for i in range(number_of_bins):
    #    for j in range(len(dict_list)):
    #        causal_effect[i][j] -= causal_means[j]
            
            
    #print(conditional_probability)
    
    for i in range(number_of_bins):

        for j in range(len(dict_list)):

            for key,val in dict_list[j].items():
                if key[layer_indices[0]+node] == i:
                    prior_sum[i][j] = prior_sum[i][j] + val

    for i in range(number_of_bins):
        for j in range(len(dict_list)):
            prior_probability[i][j] = prior_sum[i][j]/(np.sum(prior_sum[:, j]) + 10e-7)
    
    expected_causal = np.zeros((number_of_bins, len(dict_list)))
    
    for j in range(len(dict_list)):
        
        for i in range(number_of_bins):
            
            expected_causal[i][j] += np.sum(np.delete(causal_effect[:, j], i)*np.delete(prior_sum[:, j], i))/(np.sum(np.delete(prior_sum[:,j], i)) + 10e-7)
    
    average_causal_effect = np.zeros((number_of_bins, len(dict_list)))
    for i in range(number_of_bins):
        for j in range(len(dict_list)):
            average_causal_effect[i][j] = (causal_effect[i][j] - expected_causal[i][j]) * prior_probability[i][j] * 100

    return(causal_effect, average_causal_effect)

if __name__ == "__main__":
    main()
