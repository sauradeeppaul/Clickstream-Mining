import random
import pickle as pkl
import argparse
import csv
import numpy as np

import pandas as pd
import scipy as sp
import math
from scipy.stats import chisquare

'''
TreeNode represents a node in your decision tree
TreeNode can be:
    - A non-leaf node: 
        - data: contains the feature number this node is using to split the data
        - children[0]-children[4]: Each correspond to one of the values that the feature can take
        
    - A leaf node:
        - data: 'T' or 'F' 
        - children[0]-children[4]: Doesn't matter, you can leave them the same or cast to None.

'''
'''Calculates entropy'''
def get_entropy(col):
    s = sum(col)
    col = [float(p)/s for p in col]
    '''Finding Entropy'''
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in col])
    return entropy

'''Gives unique values for each column'''
def get_feature_values(features):
    uni_val = []
    for i in features.columns:
        uni_val.append((i, features[i].unique()))
    return uni_val

'''Returns feature with maximum gain'''
def get_max_gain_feature(data):
    global uni_val
    Gain = []
    for i, values in uni_val:
        neg_gain = 0
        for val in data[i].unique():
#             tdf = data.loc[data[i] == val]
            ei = get_entropy(data[i].value_counts())
            pi = float((data[i]==val).sum()) / float(data.shape[0])
            neg_gain += pi * ei
            neg_gain += ei
        Gain.append((i, neg_gain, values))

    return max(Gain, key=lambda g: g[1])

'''Calculates p value'''
def calculatePVal(X):
    (c, p) = chisquare(list(X.value_counts()))
    return p

'''ID3 decision tree building function'''
def perform_ID3(data, parent, index):
    global p
    i, gain, vals = get_max_gain_feature(data)
    node = TreeNode(i)
    parent.nodes[index] = node
    for v in vals:
        df = data.loc[data[i] == v]
        '''Recursive tree generation'''
        if calculatePVal(data[i]) > p:
            perform_ID3(df, node, v-1)
        else:
            if sp.stats.mode(df[df.shape[1]-1])[0][0] == 1:
                node.nodes[v-1] = TreeNode('T')
            else:
                node.nodes[v-1] = TreeNode('F')

# DO NOT CHANGE THIS CLASS
class TreeNode():
    def __init__(self, data='T',children=[-1]*5):
        self.nodes = list(children)
        self.data = data


    def save_tree(self,filename):
        obj = open(filename,'w')
        pkl.dump(self,obj)

# loads Train and Test data
def load_data(ftrain, ftest):
	Xtrain, Ytrain, Xtest = [],[],[]
	with open(ftrain, 'rb') as f:
	    reader = csv.reader(f)
	    for row in reader:
	        rw = map(int,row[0].split())
	        Xtrain.append(rw)

	with open(ftest, 'rb') as f:
	    reader = csv.reader(f)
	    for row in reader:
	        rw = map(int,row[0].split())
	        Xtest.append(rw)

	ftrain_label = ftrain.split('.')[0] + '_label.csv'
	with open(ftrain_label, 'rb') as f:
	    reader = csv.reader(f)
	    for row in reader:
	        rw = int(row[0])
	        Ytrain.append(rw)

	print('Data Loading: done')
	return Xtrain, Ytrain, Xtest


num_feats = 274

def create_random_tree(depth):
    if(depth >= 7):
        if(random.randint(0,1)==0):
            return TreeNode('T',[])
        else:
            return TreeNode('F',[])

    feat = random.randint(0,273)
    root = TreeNode(data=str(feat))

    for i in range(5):
        root.nodes[i] = create_random_tree(depth+1)

    return root
    
    
parser = argparse.ArgumentParser()
parser.add_argument('-p', required=True)
parser.add_argument('-f1', help='training file in csv format', required=True)
parser.add_argument('-f2', help='test file in csv format', required=True)
parser.add_argument('-o', help='output labels for the test dataset', required=True)
parser.add_argument('-t', help='output tree filename', required=True)

args = vars(parser.parse_args())

pval = args['p']
Xtrain_name = args['f1']
Ytrain_name = args['f1'].split('.')[0]+ '_label.csv' #labels filename will be the same as training file name but with _label at the end

Xtest_name = args['f2']
Ytest_predict_name = args['o']

tree_name = args['t']



Xtrain, Ytrain, Xtest = load_data(Xtrain_name, Xtest_name)

train_features = pd.read_csv(Xtrain_name, header=None, sep=" ")
train_labels = pd.read_csv(Ytrain_name, names=[train_features.shape[1]])
test_features = pd.read_csv(Xtest_name, header=None, sep=" ")


print("Training...")

# train_features = Xtrain
# train_labels = Ytrain

c = train_labels[train_features.shape[1]].value_counts()
e = get_entropy(c)

'''Finding unique values in each column'''
uni_val = get_feature_values(train_features)
'''Concatenating training labels and data'''
train_data = pd.concat([train_features, train_labels], axis=1)
'''Finding gain'''
# print train_data

p = pval

i, gain, vals = get_max_gain_feature(train_data)

tree = TreeNode(i)

for v in train_data[i].unique():
    df = train_data.loc[train_data[i] == v]
    if calculatePVal(train_data[i]) > p:
        perform_ID3(df, tree, v-1)
    else:
        if sp.stats.mode(df[df.shape[1]-1])[0][0] == 1:
            tree.nodes[v-1] = TreeNode('T')
        else:
            tree.nodes[v-1] = TreeNode('F')



# s = create_random_tree(0)
s = tree
s.save_tree(tree_name)
print("Testing...")
Ypredict = []

nodeCount = 0

def evaluate_datapoint(root,datapoint):
    global nodeCount
    nodeCount += 1
    if root.data == 'T': return 1
    if root.data =='F': return 0
    return evaluate_datapoint(root.nodes[datapoint[int(root.data)-1]-1], datapoint)


for i in range(0,len(Xtest)):
	Ypredict.append([evaluate_datapoint(tree,Xtest[i])])

print "Node Count: ", nodeCount

with open(Ytest_predict_name, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(Ypredict)

print("Output files generated")








