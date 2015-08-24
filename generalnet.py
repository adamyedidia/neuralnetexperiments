import random
import sys


# This file assumes that there's a full connection between layers at each layer

class Node:    
    # func is a function that takes in a list of inputs and a list of weights.
    # x_nextLayer = f(x_prevLayer, w)
    def __init__(self, weights, f, dFdX, dFdW, ID):
        self.weights = weights
        self.f = f
        self.dFdX = dFdX
        self.dFdW = dFdW   
        self.ID = ID
        # a list of the outputs of this node for each of the input-sets in the mini-batch     
        self.outs = None
        self.dCdOut = None
        self.dCdW = eone
            
    def computeOuts(self, listOfInputLists):
        
        # For each input-set, give the inputs from the previous layer
        self.outs = [self.f(listOfInputs, self.weights) for listOfInputs in listOfInputLists]
        
        return self.outs
        
    def computeOutDerivativeFromCost(self, trueSolutions, allNodeOuts, dCdX):
        self.dCdOut = dCdX(allNodeOuts, trueSolutions, self.ID)
        
    def computeOutDerivativeFromNextLayer(self, listOfInputs, nextLayer):
        self.dCdOut = 0
        
        for node in nextLayer.listOfNodes:
            self.dCdOut += node.dCdOut * node.dFdX(listOfInputs, node.weights, self.ID)
            # doing the chain rule out
        
    def computeWeightDerivative(self, listOfInputs):
        self.dCdW = [0] * len(self.weights)
        
        for weightIndex, weight in enumerate(self.weights):
            self.dCdW[weightIndex] += self.dCdOut * self.dFdW(listOfInputs, self.weights, weightIndex)

    def changeWeights(self, eta):
        for i in enumerate(self.weights):
            self.weights[i] -= self.dCdW[i] * eta  
            
class InputNode(Node):
    def __init__(self, outs):
        self.outs = outs
        
    def setOuts(self, newOuts):
        self.outs = newOuts      
            
class Layer:
    def __init__(self, listOfNodes, prevLayer, nextLayer):
        self.listOfNodes = listOfNodes
        self.prevLayer = prevLayer
        self.nextLayer = nextLayer
        self.outs = None
        
    def feedForward(self):
        for node in self.listOfNodes:
            node.computeOut(self.prevLayer.outs)
        
        self.outs = [[node.outs[i] for node in self.listOfNodes] for i, outs in \
                enumerate(self.listOfNodes[0].outs)])
                     
    def backPropagate(self, trueSolutions, dCdX):        
        if self.nextLayer == None:
            for node in self.listOfNodes:
                node.computeOutDerivativeFromCost(trueSolutions, [otherNode.outs for otherNode in self.listOfNodes], dCdX)
        
        else:
            for node in self.listOfNodes:
                node.computeOutDerivativeFromNextLayer(self, nextLayer)
        
        for node in self.listOfNodes:
            node.computeWeightDerivative()

    def changeWeights(self, eta):
        for node in self.listOfNodes:
            node.changeWeights(eta)
                                         
class InputLayer(Layer):
    def __init__(self, listOfNodes, nextLayer):
        self.listOfNodes = listOfNodes
        self.nextLayer = nextLayer

    def setInputs(self, inputs):
        print inputs
        print len(inputs), len(self.listOfNodes)
        assert len(inputs) == len(self.listOfNodes)
        for i, inp in enumerate(inputs):
            self.listOfNodes[i].setOuts(inp)
    
class Network:
    def __init__(self, listOfLayers, costFunc, dCdX):
        self.listOfLayers = listOfLayers
        self.costFunc = costFunc
        self.dCdX = dCdX
        
    def feedForward(self, listOfInputLists):
        self.listOfLayers[0].setInputs(listOfInputLists)
        
        for layer in self.listOfLayers[1:]:
            layer.feedForward()            
            
        return [node.outs for node in self.listOfLayers[-1].listOfNodes]
        
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        
        if test_data: 
            n_test = len(test_data)
            
        n = len(training_data)
        
        for j in xrange(epochs):
            random.shuffle(training_data)
            
#            mini_batches = [
 #               training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
 
            mini_batches = training_data[:mini_batch_size]
                
            self.backPropagate(mini_batches, eta)
            
            if test_data:
#                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
                print "Epoch {0}: cost is {1}".format(j, self.evaluate(test_data))
                
            else:
                print "Epoch {0} complete".format(j)
    
    def evaluate(self, test_data):
        return sum([self.costFunc(self.feedForward(dataPoint[0]), dataPoint[1]) for
            dataPoint in test_data])
    
    def backPropagate(self, mini_batches, eta):
        inputs = [[mini_batches[i][0][j] for i in range(len(mini_batches))] \
            for j in range(len(mini_batches[0][0]))]
        solutions = [[mini_batches[i][1][j] for i in range(len(mini_batches))] \
            for j in range(len(mini_batches[0][1]))]

        self.feedForward(inputs)

        for i in range(len(self.listOfLayers) - 1, -1, -1):
            layer = self.listOfLayers[i]
            layer.backPropagate(solutions, self.dCdX)
            
        for i in range(len(self.listOfLayers) - 1, -1, -1):
            layer = self.listOfLayers[i]
            layer.changeWeights(eta)
        
def softNand(listOfInputs, listOfWeights):
    # computes 1 - a^{w_1}b^{w_2}c^{w_3}...
    
    assert len(listOfInputs) == len(listOfWeights)
    product = 1
    
    for i in range(len(listOfInputs)):
        product *= listOfInputs[i] ** listOfWeights[i]
        
    return 1 - product
    
# differentiatedWeight is expected as an index into listOfWeights
def dSoftNanddW(listOfInputs, listOfWeights, differentiatedWeight):
    product = -1
    print listOfInputs
    print listOfWeights
    print len(listOfInputs), len(listOfWeights)
    assert len(listOfInputs) == len(listOfWeights)
    
    for i in range(len(listOfInputs)):
        product *= listOfInputs[i] ** listOfWeights[i]
    
    product *= math.log(listOfInputs[differentiatedWeight])
    
    return product    
        
def dSoftNanddX(listOfInputs, listOfWeights, weightIndex):
    product = -1
    
    assert len(listOfInputs) == len(listOfWeights)
    
    for i in range(len(listOfInputs)):
        product *= listOfInputs[i] ** listOfWeights[i]
    
    product *= listOfWeights[weightIndex] / float(listOfInputs[weightIndex])            
        
def crossEntropyCost(proposedSolutions, trueSolutions):
    numOutputs = len(proposedSolutions)
    
    if numOutputs > 0:
        batchSize = len(proposedSolutions[0])
        
    overallSum = 0
    
    for i in range(numOutputs):
        for j in range(batchSize):
            y = trueSolutions[i][j]
            a = proposedSolutions[i][j]
            overallSum += y * math.log(a) + (1 - y) * math.log(1 - a)
        
    overallSum /= numOutputs * batchSize
    
    return overallSum
        
def crossEntropyCostPrime(proposedSolutions, trueSolutions, outputIndex):
    numOutputs = len(proposedSolutions)
    
    print proposedSolutions

    if numOutputs > 0:
        batchSize = len(proposedSolutions[0])

    overallSum = 0
    
    print len(proposedSolutions)
    print len(proposedSolutions[0])
    print len(trueSolutions)
    print len(trueSolutions[0])
    print trueSolutions

    for j in range(batchSize):
        overallSum += proposedSolutions[outputIndex][j] / trueSolutions[outputIndex][j]
        overallSum -= (1 - proposedSolutions[outputIndex][j]) / \
            (1 - trueSolutions[outputIndex][j])

    return overallSum    
        
def allSizeXLists(x):
    if x == 0:
        return [[]]
        
    allSizeXMinus1Lists = allSizeXLists(x - 1)    
        
    return [littleList + [0.1] for littleList in allSizeXMinus1Lists] + \
        [littleList + [0.9] for littleList in allSizeXMinus1Lists]

trainingList = []

for size4List in allSizeXLists(4):        
    trainingExample = (size4List, [0.8*(sum(size4List) < 1.3)+0.1, 0.8*(sum(size4List) == 2.0)+0.1, 
                        0.8*(sum(size4List) == 2.8)+0.1, 0.8*(sum(size4List) == 3.6)+0.1])
    
    trainingList += [trainingExample] * 10
    
    
                                
nodesPerLayer = int(sys.argv[1])
numLayers = int(sys.argv[2])    
        
inputLayer = InputLayer([InputNode(0) for i in range(nodesPerLayer)], None)
listOfLayers = [inputLayer]

for i in range(numLayers - 1):
    listOfNodesInLayer = [Node([1./nodesPerLayer for i in range(nodesPerLayer)], softNand, \
            dSoftNanddX, dSoftNanddW, i) for i in range(nodesPerLayer)]
    layer = Layer(listOfNodesInLayer, listOfLayers[-1], None)
    listOfLayers.append(layer)
    
for i in range(numLayers - 1):
    listOfLayers[i].nextLayer = listOfLayers[i+1]    
    
network = Network(listOfLayers, crossEntropyCost, crossEntropyCostPrime)    

print network.feedForward([[0.1, 0], [1, 0], [1, 1], [1, 1]])
#print trainingList
network.SGD(trainingList, 30, 10, 0.1)    
