import random
import sys
import math

def EPS():
    return 0.001

class NodeFunctionDescription:
    def __init__(self, f, dFdX, dFdW):
        self.f = f
        self.dFdX = dFdX
        self.dFdW = dFdW
        
class CostFunctionDescription:
    def __init__(self, costFunc, dCdX):
        self.costFunc = costFunc
        self.dCdX = dCdX

class Node:
    def __init__(self, weights, f, dFdX, dFdW, ID, numInputs):
        self.weights = weights
              
        # UNCHANGING
        self.f = f
        self.dFdX = dFdX
        self.dFdW = dFdW
        self.ID = ID    
            
        # Temporary memory for dynamic programming
        self.out = None
        self.dCdOut = None
        self.dCdW = [0] * numInputs

    def computeOut(self, inputs):
        self.out = self.f(inputs, self.weights)
        
        return self.out
        
    def computeOutDerivativeFromCost(self, myOutput, solution, dCdX):
        self.dCdOut = dCdX(myOutput, solution)
#        print "cost1", crossEntropyCost(thisLayerOutputs, trueSolutions)
        
#        thisLayerOutputsCopy = thisLayerOutputs[:]
#        thisLayerOutputsCopy[self.layerIndex] += 0.000001
        
#        print "cost2", crossEntropyCost(thisLayerOutputsCopy, trueSolutions)
        
#        print "cost1-cost2", crossEntropyCost(thisLayerOutputs, trueSolutions) - \
#         crossEntropyCost(thisLayerOutputsCopy, trueSolutions)
         
#        assert abs(crossEntropyCost(thisLayerOutputs, trueSolutions) - \
#            crossEntropyCost(thisLayerOutputsCopy, trueSolutions)) < 0.01
        
#        print self.layerIndex
#        print "diff", crossEntropyCostPrime(thisLayerOutputs, trueSolutions, self.layerIndex)
        
#        print "ratio", crossEntropyCostPrime(thisLayerOutputs, trueSolutions, self.layerIndex) / (crossEntropyCost(thisLayerOutputs, trueSolutions) - \
#         crossEntropyCost(thisLayerOutputsCopy, trueSolutions))
    
    def computeOutDerivativeFromLaterNodes(self, allInputs, laterNodes):
        self.dCdOut = 0
        
        for node in laterNodes:
#            print node.dCdOut
#            print node.ID, len(node.weights)
#            print allInputs
            self.dCdOut += node.dCdOut * node.dFdX(allInputs[:len(node.weights)], node.weights, 
                self.ID)
                
#            print self.ID, self.dCdOut, "dCdOut"
            
                        
#        x1 = node.f(thisLayerOutputs, node.weights)
        
#        thisLayerOutputsCopy = thisLayerOutputs[:]
#        thisLayerOutputsCopy[self.layerIndex] += 0.000001
        
#        x2 = node.f(thisLayerOutputsCopy, node.weights)
        
#        assert abs(x1 - x2) < 0.01
        
#        print "diffx", (x1 - x2) * 1000000 
        
#        print "diffx", node.dFdX(thisLayerOutputs, node.weights, self.layerIndex)
        
        
                
    def computeWeightDerivative(self, prevNodeOutputs):
 #       self.dCdW = [0] * len(self.weights)
        
        for weightIndex, weight in enumerate(self.weights):
#            print self.dFdW(prevNodeOutputs, self.weights, weightIndex), "dFdW"
            
#            print "dCdOut", self.dCdOut, "dFdW", self.dFdW(prevNodeOutputs, self.weights, weightIndex)
            
            self.dCdW[weightIndex] += self.dCdOut * self.dFdW(prevNodeOutputs, self.weights, weightIndex)
            
#            print self.dCdW, "dCdW"
            
#            w1 = self.f(prevLayerOutputs, self.weights)
            
#            selfWeightsCopy = self.weights[:]
#            selfWeightsCopy[weightIndex] += 0.000001
            
#            w2 = self.f(prevLayerOutputs, selfWeightsCopy)
            
#            assert abs(w1 - w2) < 0.01
            
#            print "diffy", (w1 - w2) * 1000000
            
#            print "diffy", self.dFdW(prevLayerOutputs, self.weights, weightIndex)
            
    def changeWeights(self, eta):
        for i in range(len(self.weights)):
            # Weights cannot reach 0 in the NAND-net
            self.weights[i] = min(max(EPS(), self.weights[i] - self.dCdW[i] * eta), 1-EPS())
        
#            self.weights[i] = self.weights[i] - self.dCdW[i] * eta
            
#            print self.ID, self.weights[i], self.dCdW[i] * eta  
        
#        print self.ID, self.dCdOut, "dCdOut"      
        
        print self.weights
        
        self.dCdW = [0] * len(self.weights)

class DAG:
    
    # A DAG in which all the nodes evaluate the same function
    def __init__(self, numInputs, numNodes, nfd, cfd):
        
        self.listOfNodes = []
        
        for numIncidentNodes in range(numInputs, numNodes+numInputs):
#            weights = [1./numIncidentNodes + random.random()*0.1 for i in range(numIncidentNodes)]
            
            
            # These are explicitly setting up a XOR gate made out of NANDs.
            
            EPS = 0.4
            
            if numIncidentNodes == 2:
#                weights = [1-EPS(), 1-EPS()]
                weights = [1-EPS, 1-EPS]
                
            if numIncidentNodes == 3:
#                weights = [1-EPS(), EPS(), 1-EPS()]
                weights = [1-EPS, EPS, 1-EPS]
            
            if numIncidentNodes == 4:
#                weights = [EPS(), 1-EPS(), 1-EPS(), EPS()]
                weights = [EPS, 1-EPS, 1-EPS, EPS]
                
            if numIncidentNodes == 5:
#                weights = [EPS(), EPS(), EPS(), 1-EPS(), 1-EPS()]
                weights = [EPS, EPS, EPS, 1-EPS, 1-EPS]
            
            newNode = Node(weights, nfd.f, nfd.dFdX, nfd.dFdW, numIncidentNodes - numInputs, numIncidentNodes)
            
            self.listOfNodes.append(newNode)
    
        self.costFunc = cfd.costFunc
        self.dCdX = cfd.dCdX
        
    def feedForward(self, inputs):
        # This is the input to each of the next nodes
        runningInput = inputs[:]
        
        for node in self.listOfNodes:
            nodeOutput = node.computeOut(runningInput)
            
            runningInput.append(nodeOutput)
        
        return runningInput
            
    def batchGradientDescent(self, batch, epochs, eta, testData=None):
        for j in range(epochs):
            self.batchBackPropagate(batch, eta)
            
            if testData:
                print "Epoch {0}: cost is {1}".format(j, self.evaluate(testData))
                
            else:
                print "Epoch {0} complete".format(j)
                
    def batchBackPropagate(self, batch, eta):
        for dataPoint in batch:
            inputs = dataPoint[0]
            solution = dataPoint[1]
            
            # This is all the inputs that could be fed into a node--including the outputs of nodes
            allInputs = self.feedForward(inputs)
            
            # For the time being, I will assume only the final node is an output node
            outputNode = self.listOfNodes[-1]
            
            outputNode.computeOutDerivativeFromCost(outputNode.out, solution, self.dCdX)
            outputNode.computeWeightDerivative(allInputs[:len(outputNode.weights)])
            
            for i in range(len(self.listOfNodes) - 2, -1, -1):
                node = self.listOfNodes[i]
                node.computeOutDerivativeFromLaterNodes(allInputs, self.listOfNodes[i+1:])
            
                node.computeWeightDerivative(allInputs[:len(node.weights)])
                
        self.changeWeights(eta)
        
    def changeWeights(self, eta):
        for node in self.listOfNodes:
            node.changeWeights(eta)
    
def softNand(inputs, weights):
    assert len(inputs) == len(weights)
    product = 1
    
#    print inputs
    
    for i in range(len(inputs)):
        product *= inputs[i] ** weights[i]
        
    return 1 - product
    
def dSoftNanddW(inputs, weights, weightIndex):
    product = -1
    
    assert len(inputs) == len(weights)
    
    for i in range(len(inputs)):
        product *= inputs[i] ** weights[i]
        
    product *= math.log(inputs[weightIndex])
    
    return product
    
def dSoftNanddX(inputs, weights, inputIndex):
    product = -1
    
    assert len(inputs) == len(weights)
    
    for i in range(len(inputs)):
        product *= inputs[i] ** weights[i]
        
    product *= weights[inputIndex] / inputs[inputIndex]
    
    return product
        
def allSizeXLists(x):
    if x == 0:
        return [[]]
        
    allSizeXMinus1Lists = allSizeXLists(x - 1)
    
    return [littleList + [EPS()] for littleList in allSizeXMinus1Lists] + \
        [littleList + [1-EPS()] for littleList in allSizeXMinus1Lists]
        
def crossEntropyCost(output, solution):
    return -1 * (solution * math.log(output) + (1 - solution) * math.log(1 - output))
    
def crossEntropyCostPrime(output, solution):  
    return -1 * (solution / output - (1 - solution) / (1 - output))

nfd = NodeFunctionDescription(softNand, dSoftNanddX, dSoftNanddW)
cfd = CostFunctionDescription(crossEntropyCost, crossEntropyCostPrime) 
        
numNodes = int(sys.argv[1])        
        
dag = DAG(2, numNodes, nfd, cfd)

trainingList = []
templateList = []

allSizeXLists2 = allSizeXLists(2)

#allSizeXList2 = [[0.1, 0.9]]

for size2List in allSizeXLists2:
    if ((size2List[0] == 1-EPS()) ^ (size2List[1] == 1-EPS())):
        trainingExample = [size2List, 1-EPS()]
    else:
        trainingExample = [size2List, EPS()]
    
    trainingList += [trainingExample] * 10
    templateList += [trainingExample]

first = dag.feedForward([EPS(), EPS()])[-1]
second = dag.feedForward([EPS(), 1-EPS()])[-1]
third = dag.feedForward([1-EPS(), EPS()])[-1]
fourth = dag.feedForward([1-EPS(), 1-EPS()])[-1] 
                            
dag.batchGradientDescent(templateList, int(sys.argv[2]), 0.05)                            
    
print first, EPS(), crossEntropyCost(first, EPS())
print second, 1-EPS(), crossEntropyCost(second, 1-EPS())
print third, 1-EPS(), crossEntropyCost(third, 1-EPS())
print fourth, EPS(), crossEntropyCost(fourth, EPS())

print ''

print dag.feedForward([EPS(), EPS()])[-1], EPS(), crossEntropyCost(dag.feedForward([EPS(), EPS()])[-1], EPS())
print dag.feedForward([EPS(), 1-EPS()])[-1], 1-EPS(), crossEntropyCost(dag.feedForward([EPS(), 1-EPS()])[-1], 1-EPS())
print dag.feedForward([1-EPS(), EPS()])[-1], 1-EPS(), crossEntropyCost(dag.feedForward([1-EPS(), EPS()])[-1], 1-EPS())
print dag.feedForward([1-EPS(), 1-EPS()])[-1], EPS(), crossEntropyCost(dag.feedForward([1-EPS(), 1-EPS()])[-1], EPS())    