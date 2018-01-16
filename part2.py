# -*- coding: utf-8 -*-
from sklearn.neighbors import KNeighborsClassifier
from sfs import sfs
import numpy as np

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
    #prepare parcite and test sets
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
    index_group = sfs(practice_set,y,8,clf2,score)
    np_array_convertor = [np.array(r) for r in practice_set]
    practice_set = [r[index_group] for r in np_array_convertor]
    clf2.fit(practice_set,y)
    np_array_convertor = [np.array(r) for r in test_set]
    test_set = [r[index_group] for r in np_array_convertor]
    print(clf2.score(test_set,answers))
if __name__ == '__main__':
    main()
