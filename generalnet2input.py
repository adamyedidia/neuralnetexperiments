import random

class InputNode:
    def __init__(self, value):
        self.value = value
        
    def computeOut(self):
        return self.value
    

class Node:
    
    # func must be a function with three arguments, the first of which is the "weight" argument
    def __init__(self, func, funcPrime, weight=random.random()):
        self.weight = weight
        self.func = func
        self.funcPrime = funcPrime
    
    def setInputs(self, node1, node2):
        self.input1 = node1
        self.input2 = node2
            
    def computeOut(self):
        return self.func(self.weight, self.input1.out, self.input2.out)
        
    
    
class Layer:
    def __init__(self, listOfNodes):
        self.listOfNodes = listOfNodes
        
    def adjacencyConnection(self, otherLayer, step):
        for node in self.listOfNodes:
            
    
class Network:
    def __init__(self, listOfLayers):