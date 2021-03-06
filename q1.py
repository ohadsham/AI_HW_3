# -*- coding: utf-8 -*-
from numpy import log2,inf
#general:
#each element represent as line of the file.
#all features have binary values (true or false) 

# convert feature name to index in line
def featureToIndex(feature):
    return attributes.index(feature)

def makeTree(examples,features,default):
    if not examples :
        return ([],[],default)
    c = majorityClass(examples)
    #features == ['classification'] means features list is empty
    if c[1] or features == ['classification']:
        return ([],[],c[0])
    c = c[0]
    f = selectFeature(features,examples) #f <- SelectFeature(Features, E)
    F = features.copy()
    F.remove(f) #F <- Features-{f}
    
    f_index = featureToIndex(f)
    # feature value can be 1 or 0. we create two sub trees. one with all the element with 
    #0 value second with all the element with 1 value
    subtrees = ('0',makeTree([x for x in examples if x[f_index] =='0'],F,c)),('1',makeTree([x for x in examples if x[f_index] =='1'],F,c))
    return (f,subtrees,c)
  
def selectFeature(features,examples):
    if features[0] == 'classification':
        return []   
    return minEntropyOfFeature(features,examples)

def classify(element,tree):
    (feature,childrens,value) = tree
    if childrens == []:
        return value
    v = element[featureToIndex(feature)]
    if v == '0':
        subtree = ((childrens[0])[1])
    else:
        subtree = ((childrens[1])[1])
    return classify(element,subtree)
    
#return true if most of the  elements are in the "true" class
#otherwise return false
# return value tuple: (majority class,isAllAgree)
#if all elements are in the same class then "isAllAgree" is true
def majorityClass(examples):
    total_false = 0
    total_true = 0
    isAllAgree = False
    classIndex = len(examples[0]) - 1
    
    for element in examples:
        if element[classIndex] == 'True':
            total_true+=1
        else:
            total_false+=1
    if total_true==0 or total_false==0:
        isAllAgree=True
    return (total_true>total_false,isAllAgree)

def entropy(examples):
    total_false = 0
    total_true = 0
    if(not examples):
        return 0
    classIndex = len(examples[0])-1
    
    for element in examples:
        if element[classIndex] == 'True':
            total_true+=1
        else:
            total_false+=1
    total = total_false + total_true
    true_entropy =total_true/total
    false_entropy =total_false/total
    if total_true==0:
       return -1*(false_entropy)*log2((false_entropy)) 
    if total_false==0:
        return -1*(true_entropy)*log2((true_entropy))       
    return -1*(false_entropy)*log2((false_entropy))+ -1*(true_entropy)*log2((true_entropy))

#for a group of features and examples we iterate over the features and caculate for 
#each feature the entropy of the two divide group. we return the feature with 
#the minimum entropy. (=means the information gain is the largest)
def minEntropyOfFeature(features,examples):
    f_min =-1
    minEnt = inf
    lLen = len(examples)
    for f in features:
    #end of features
      if f == 'classification':
          return f_min
      f_index = featureToIndex(f)
      f0_divide = [x for x in examples if x[f_index] =='0']
      f1_divide = [x for x in examples if x[f_index] =='1']
      current_entropy =(len(f0_divide)/lLen) *entropy(f0_divide)+(len(f1_divide)/lLen) *entropy(f1_divide)
      if current_entropy < minEnt:
         minEnt = current_entropy
         f_min = f
    return f_min

#returns successeful qualifications number
def hitRatio(tree,examples):
    hitCounter = 0
    for example in examples:
        if str(classify(example,tree)) == example[len(example)-1]:
            hitCounter+=1
    return hitCounter
        
#calc average accuracy of 4fold examples 
def calcAcc(examples):
    total_acc = 0
    fold_size = len(examples)/4
    
    for x in range(4):
        fold3 = examples[0:round(x*fold_size)]+examples[round((x+1)*fold_size):]  
        fold1 = examples[round(x*fold_size):round((x+1)*fold_size)]
        tree  = makeTree(fold3,attributes,True)
        total_acc += hitRatio(tree,fold1)
    return total_acc/len(examples)    

#calc confuison matrix 4fold examples 
def calcConfusionMatrix(examples):
    fold_size = len(examples)/4
    true_pos = 0
    false_pos = 0
    false_neg = 0
    true_neg = 0
    for x in range(4):
        fold3 = examples[0:round(x*fold_size)]+examples[round((x+1)*fold_size):]  
        fold1 = examples[round(x*fold_size):round((x+1)*fold_size)]
        tree  = makeTree(fold3,attributes,True)
        for example in fold1:
            if str(classify(example,tree)) == example[len(example)-1]:
                if example[len(example)-1] == 'True':
                    true_pos+=1
                else:
                    true_neg+=1
            else:
                if example[len(example)-1] == 'False':
                    false_pos+=1
                else:
                    false_neg+=1 
    print("[["+str(true_pos)+" "+str(false_pos)+"]")
    print("[ "+str(false_neg)+" "+str(true_neg)+"]]")
    
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
    print(calcAcc(examples))
    calcConfusionMatrix(examples)
    file.close()
	
if __name__ == '__main__':
    main()