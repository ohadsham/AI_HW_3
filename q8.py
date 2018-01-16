# -*- coding: utf-8 -*-

# convert feature name to index in line
def featureToIndex(feature):
    return attributes.index(feature)

def makeTree(examples,features,default,size_param):
    if not examples or len(examples)<size_param :
        return ([],[],default)
    c = majorityClass(examples)
    #features == ['classification'] means features list is empty
    if c[1] or features == ['classification']:
        return ([],[],c[0])
    c = c[0]
    f = selectFeature(features,examples) #f <- SelectFeature(Features, E)
    F = features.copy()
    if (f not in F):
        print(f)
    F.remove(f) #F <- Features-{f}
    
    f_index = featureToIndex(f)
    # feature value can be 1 or 0. we create two sub trees. one with all the element with 
    #0 value second with all the element with 1 value
    subtrees = ('0',makeTree([x for x in examples if x[f_index] =='0'],F,c,size_param)),('1',makeTree([x for x in examples if x[f_index] =='1'],F,c,size_param))
    return (f,subtrees,c)
  
def selectFeature(features,examples):
    if features[0] == 'classification':
        return []   
    return features[len(features)-2]

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

#returns successeful qualifications number
def hitRatio(tree,examples):
    hitCounter = 0
    for example in examples:
        if str(classify(example,tree)) == example[len(example)-1]:
            hitCounter+=1
    return hitCounter

#calc average accuracy of  examples 
def calcAcc(tree,examples):    
    return hitRatio(tree,examples)/len(examples)
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
    practice_set = examples[int(len(examples)/4):]
    test_set = examples[0:int(len(examples)/4)]
    tree = makeTree(practice_set,attributes,True,0)
    print(calcAcc(tree,test_set))
    tree = makeTree(practice_set,attributes,True,20)
    print(calcAcc(tree,test_set))
    file.close()
if __name__ == '__main__':
    main()

