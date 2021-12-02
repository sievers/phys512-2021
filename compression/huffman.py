import numpy as np
import copy

#class for nodes in a tree
class node:
    def __init__(self,name=None,p=0):
        self.name=name
        self.left=None
        self.right=None
        self.prob=p
    def print(self,cur=''):
        if (self.left is None)&(self.right is None):
            print(self.name,' ',cur)
        else:
            self.left.print(cur+'0')
            self.right.print(cur+'1')
    def get_length(self,depth=0):
        if (self.left is None)&(self.right is None):
            #s=-self.prob*np.log2(self.prob)
            return self.prob*depth
        else:
            s=0
            s=s+self.left.get_length(depth+1)
            s=s+self.right.get_length(depth+1)
        return s
def merge_nodes(probs,nodes):
    ind=np.argmin(probs)
    node1=nodes[ind]
    p1=probs[ind]
    del(probs[ind])
    del(nodes[ind])

    ind=np.argmin(probs)
    node2=nodes[ind]
    p2=probs[ind]
    del(probs[ind])
    del(nodes[ind])
    mynode=node()
    mynode.left=node1
    mynode.right=node2
    myprob=p1+p2
    probs.append(myprob)
    nodes.append(mynode)
    return probs,nodes


def build_tree(names,probs):
    names=copy.copy(names)
    probs=copy.copy(probs)
    n=len(names)
    ptot=np.sum(probs)
    nodes=[None]*len(names)
    for i in range(n):
        nodes[i]=node(names[i],probs[i]/ptot)
    for i in range(n-1):
        probs,nodes=merge_nodes(probs,nodes)
    return nodes[0]

#names=['a','b','c']
#probs=[0.4,0.25,0.35]
#names=['a','c','e','d','b']
#probs=[7,6,5,2,1]
names=['a','b','c','d','e','f','g']
probs=[1,1,2,4,8,8,8]
tree=build_tree(names,probs)


print('average bits per character: ',tree.get_length())
pp=np.asarray(probs,dtype='float')
pp=pp/pp.sum()
print('ideal would have been ',-np.sum(pp*np.log2(pp)))
