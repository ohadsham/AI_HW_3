# -*- coding: utf-8 -*-
from sklearn.neighbors import KNeighborsClassifier
from sfs import sfs
import numpy as np
import q8

def score(clf,x,y):
    clf.fit(x,y)
    return clf.score(x,y)
    

def main():
    #===start parse data
    file = open ('flare.csv')
    examples = [[]]
    for line in file:
        line = line.strip("\r\n")
        examples.append(line.split(','))
    examples.remove([])
    global attributes
    attributes = examples[0]
    examples.remove(attributes)
    #===end parse data
    #prepare practice and test sets
    practice_set = examples[int(len(examples)/4):]
    test_set = examples[0:int(len(examples)/4)]
    #prepare params for KNeighborsClassifier
    y = [x[len(x)-1] for x in practice_set]
    answers = [ x[len(x)-1:] for x in test_set]
    practice_set = [ x[0:len(x)-1] for x in practice_set]
    test_set = [ x[0:len(x)-1] for x in test_set]
    
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(practice_set, y)
    
    #q7p1 answer:
    print(neigh.score(test_set,answers))
    
    #q7p2
    clf2 = KNeighborsClassifier(n_neighbors=5)
    # choose the features using sfs:
    choosen_features = sfs(practice_set,y,8,clf2,score)
    
    # take only the chosen features of the samples:
    np_array_convertor = [np.array(r) for r in practice_set]
    practice_set = [r[choosen_features] for r in np_array_convertor]
    np_array_convertor = [np.array(r) for r in test_set]
    test_set = [r[choosen_features] for r in np_array_convertor]
    
    # train and return the accuracy rate with the chosen features:
    clf2.fit(practice_set,y)
    print(clf2.score(test_set,answers))
    
	# print the accuracy rates of question 8:
    q8.main()
	
if __name__ == '__main__':
    main()
