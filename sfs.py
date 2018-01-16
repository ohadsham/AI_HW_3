from  math import inf
import numpy as np

def sfs_inner(x, y, k, clf, score,feature_subset):
    
    if k==0:
        return feature_subset
    
    current_val = -inf
    index = -1
    for i in range(len(x[0])):
        if i in feature_subset:
            continue
        
        x_np = [np.array(r) for r in x]
        x_features_subset = [r[feature_subset+[i]] for r in x_np]
        
        score_rank = score(clf,x_features_subset,y)
        if  score_rank> current_val:
            current_val = score_rank
            index = i
    return sfs_inner(x,y,k-1,clf,score,feature_subset+[index])

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
