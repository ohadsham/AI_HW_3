from  math import inf

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
