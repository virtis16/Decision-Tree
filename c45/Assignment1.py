#import pdb
import pandas as pd
from random import randrange
import random
import numpy as np
from c45 import C45

def saveContent(train, filename):
        file = open(filename,"w")
        rx,cx = train.shape
        for i in range(0,rx):
            for j in range(0,cx):
                file.write(str(train[i][j]))
                if(j<cx-1):
                    file.write(',')
            file.write("\n")
        file.close()
    
def kFoldValidation(dataset, folds=3):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index)) 
		dataset_split.append(fold)
	return dataset_split

def kfold_c45(dataset, k):
    folds = np.array(kFoldValidation(dataset, k))
    rx,cx = dataset.shape
    sum = 0
    for i in range(0,k):
        train_data = np.zeros((1,cx))
        test_data = folds[i]
        for j in range(0,k):
            if i!=j:
                train_data = np.concatenate((train_data, folds[j]))
        train_data = train_data[1:,:]
        saveContent(train_data, "../data/iris/train_data")    
        c1 = C45("../data/iris/train_data", "../data/iris/iris.names")
        c1.retrieveData()
        c1.processData()
        c1.constructTree()
        print("Tree using Kfold: ")
        c1.printTree()
        accuracy = c1.accuracy(test_data)
        print(accuracy)
        sum += accuracy
    average_accuracy = sum/k
    print("Average accuracy : " , average_accuracy) 

def dataSplitting(dataset):
    rx,cx = dataset.shape
    train_data = dataset[:int((len(dataset)+1)*.80)] #Remaining 80% to training set
    test_data = dataset[int((len(dataset)+1)*.80+1):] #Splits 20% data to test set
    sum = 0
    saveContent(train_data, "../data/iris/train_data")
    c1 = C45("../data/iris/train_data", "../data/iris/iris.names")
    c1.retrieveData()
    c1.processData()
    c1.constructTree()
    print("Tree using Data Splitting:")
    c1.printTree()
    print("Accuracy: ", c1.accuracy(test_data))
    
dataset = pd.read_csv('../data/iris/iris.data', header = -1).as_matrix()# test cross validation split
k = 5
random.shuffle(dataset)
kfold_c45(dataset,k)
dataSplitting(dataset)