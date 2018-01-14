# -*- coding: utf-8 -*-
from sklearn.neighbors import KNeighborsClassifier


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
    counter = 0
    #check accuracy ratio
    for i in range(len(test_set)):
        if answers[i] == neigh.predict([test_set[i]]):
            counter+=1
    #q7 answer:
    print(counter/len(test_set))
    
    
    
    
if __name__ == '__main__':
    main()
