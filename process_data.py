# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 10:04:19 2016

@author: sanjeev
"""
from sklearn.preprocessing import LabelEncoder
def encode(train, test):
    #encode based on the species
    labelEncode = LabelEncoder().fit(train.species) 
    labels = labelEncode.transform(train.species)
    
    #get the classes    
    classes = list(labelEncode.classes_)
    test_ids = test.id
    
    #get rid of id and name of species
    train_data = train.drop(['species', 'id'], axis=1)
    
    #get rid of ids
    test_data = test.drop(['id'], axis=1)
    
    return train_data, labels, test_data, test_ids, classes