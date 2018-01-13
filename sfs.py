from  math import inf
from sklearn.neighbors import KNeighborsClassifier

def sfs_inner(x, y, k, clf, score,feature_subset):
    
    if k==0:
        return feature_subset
    
    current_val = -inf
    index = -1
    for i in range(len(x)):
        if score(clf,feature_subset+x[i],y) > current_val:
            current_val = score(clf,feature_subset+x[i],y)
            index = i
    return sfs_inner(x.remove(x[index]),y,k-1,clf,score,feature_subset+x[index])

def sfs(x, y, k, clf, score):
    """
    :param x: feature set to be trained using clf. list of lists.
    :param y: labels corresponding to x. list.
    :param k: number of features to select. int
    :param clf: classifier to be trained on the feature subset.
    :param score: utility function for the algorithm, that receives clf, feature subset and labeles, returns a score. 
    :return: list of chosen feature indexes
    """
    return sfs_inner(x, y, k, clf, score,[])
    

def main():
    file = open ('flare.csv')
    examples = [[]]
    for line in file:
        line = line.strip("\r\n")
        examples.append(line.split(','))
    examples.remove([])
    global attributes
    attributes = examples[0]
    examples.remove(attributes)
    y = [x[len(x)-1] for x in examples]
    answers = [ x[len(x)-1:] for x in examples]
    examples = [ x[0:len(x)-1] for x in examples]
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(examples, y)
    counter = 0
    for i in range(len(examples)):
        if answers[i] == neigh.predict([examples[i]]):
            counter+=1
    print(counter/len(examples))
    
    
    
if __name__ == '__main__':
    main()