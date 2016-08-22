#!/usr/bin/python

import numpy as np
import pandas as pd

def euclidean_distance(training, test):
    
    training_data_values = training
    n = len(training_data_values)
    test_data = test
    
    return [np.sqrt(sum(pow((training_data_values[i] - test_data),2))) for i in range(n)]
       
def main():

    # number of nearest neighbors to select
    k  = 3
    
    # The test data contains the length and width of the sepal and the pestal
    # the expected output is known to be Iris versicolor(https//en.wikipedia.org/wiki/Iris_flower_data_set)
    # the measurements of the object can be changed in order to classify a new object 
    test_data = np.array([5.7,2.8,4.1,1.3])
    
    # column headers
    full_header = ['sepal_length', 'sepal_width', 'pestal_length', 'pestal_width', 'label']
    sub_header = ['sepal_length', 'sepal_width', 'pestal_length', 'pestal_width']
    
    # import data into a data frame
    data = pd.read_csv('iris.training.data.txt', names = full_header)
    
    # extract values for array operations
    training_data = data[sub_header].values
    
    # extract distance using euclidean distance measure
    distance = euclidean_distance(training_data, test_data)
    
    # insert distance into data frame
    dist = pd.DataFrame(distance, columns = ['distance'])
    dataset = data.join(dist)
    
    # sort data in ascending order by distance
    sorted_data = dataset.sort(['distance'], ascending = True)
    nearest_neighbors = sorted_data._slice(slice(0,k),0)
    
    # select the label in k group that occurs with the most frequency
    classification = nearest_neighbors['label'].mode().values[0]
    
    print "The features in the test data belong to", classification
    
main()
    




