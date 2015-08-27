import random
import sys
import math

class Node:
    def __init__(self, weights, f, dFdX, dFdW, layerIndex, numInputs=int(sys.argv[1])):
        self.weights = weights
        
       
        # UNCHANGING
        self.f = f
        self.dFdX = dFdX
        self.dFdW = dFdW
        self.layerIndex = layerIndex    
            
        # Temporary memory for dynamic programming
        self.out = None
        self.dCdOut = None
        self.dCdW = [0] * numInputs

    def computeOut(self, prevLayerOutputs):
        self.out = self.f(prevLayerOutputs, self.weights)
        
        return self.out
        
    def computeOutDerivativeFromCost(self, thisLayerOutputs, trueSolutions, dCdX):
        self.dCdOut = dCdX(thisLayerOutputs, trueSolutions, self.layerIndex)
        
        print "cost1", crossEntropyCost(thisLayerOutputs, trueSolutions)
        
        thisLayerOutputsCopy = thisLayerOutputs[:]
        thisLayerOutputsCopy[self.layerIndex] += 0.000001
        
        print "cost2", crossEntropyCost(thisLayerOutputsCopy, trueSolutions)
        
#        print "cost1-cost2", crossEntropyCost(thisLayerOutputs, trueSolutions) - \
#         crossEntropyCost(thisLayerOutputsCopy, trueSolutions)
         
        assert abs(crossEntropyCost(thisLayerOutputs, trueSolutions) - \
            crossEntropyCost(thisLayerOutputsCopy, trueSolutions)) < 0.01
        
        print self.layerIndex
        print "diff", crossEntropyCostPrime(thisLayerOutputs, trueSolutions, self.layerIndex)
        
        print "ratio", crossEntropyCostPrime(thisLayerOutputs, trueSolutions, self.layerIndex) / (crossEntropyCost(thisLayerOutputs, trueSolutions) - \
         crossEntropyCost(thisLayerOutputsCopy, trueSolutions))
    
    def computeOutDerivativeFromNextLayer(self, thisLayerOutputs, nextLayer):
        self.dCdOut = 0
        
        for node in nextLayer.listOfNodes:
            print node.dCdOut
            self.dCdOut += node.dCdOut * node.dFdX(thisLayerOutputs, node.weights, 
                self.layerIndex)
                
                        
        x1 = node.f(thisLayerOutputs, node.weights)
        
        thisLayerOutputsCopy = thisLayerOutputs[:]
        thisLayerOutputsCopy[self.layerIndex] += 0.000001
        
        x2 = node.f(thisLayerOutputsCopy, node.weights)
        
        assert abs(x1 - x2) < 0.01
        
#        print "diffx", (x1 - x2) * 1000000 
        
#        print "diffx", node.dFdX(thisLayerOutputs, node.weights, self.layerIndex)
        
        
                
    def computeWeightDerivative(self, prevLayerOutputs):
 #       self.dCdW = [0] * len(self.weights)
        
        for weightIndex, weight in enumerate(self.weights):
            self.dCdW[weightIndex] += self.dCdOut * self.dFdW(prevLayerOutputs, self.weights, weightIndex)
            
            w1 = self.f(prevLayerOutputs, self.weights)
            
            selfWeightsCopy = self.weights[:]
            selfWeightsCopy[weightIndex] += 0.000001
            
            w2 = self.f(prevLayerOutputs, selfWeightsCopy)
            
            assert abs(w1 - w2) < 0.01
            
#            print "diffy", (w1 - w2) * 1000000
            
#            print "diffy", self.dFdW(prevLayerOutputs, self.weights, weightIndex)
            
    def changeWeights(self, eta):
        for i in range(len(self.weights)):
            # Weights cannot reach 0 in the NAND-net
            self.weights[i] = max(0.0001, self.weights[i] - self.dCdW[i] * eta)
            
        self.dCdW = [0] * len(self.weights)
                        
class InputNode(Node):
    def __init__(self, layerIndex):
        self.layerIndex = layerIndex
        self.dCdOut = None
        
    def setOut(self, newOut):
        self.out = newOut
        
class Layer:
    def __init__(self, listOfNodes, prevLayer, nextLayer, networkIndex):
        # UNCHANGING
        self.listOfNodes = listOfNodes
        self.prevLayer = prevLayer
        self.nextLayer = nextLayer
        self.networkIndex = networkIndex
        
        # CHANGING
        self.outputs = None
        
        
    def feedForward(self):
        for node in self.listOfNodes:
            node.computeOut(self.prevLayer.outputs)
            
        self.outputs = [node.out for node in self.listOfNodes]
        
    def backPropagate(self, solutions, dCdX):
        if self.nextLayer == None:
            # Then this is the output layer
            for node in self.listOfNodes:
                node.computeOutDerivativeFromCost(self.outputs, solutions, dCdX)
                
        else:
            for node in self.listOfNodes:
                node.computeOutDerivativeFromNextLayer(self.outputs, self.nextLayer)
        
        for node in self.listOfNodes:
            node.computeWeightDerivative(self.prevLayer.outputs)
    
    def changeWeights(self, eta):
        for node in self.listOfNodes:
            node.changeWeights(eta)
            
class InputLayer(Layer):
    def __init__(self, listOfNodes, nextLayer, networkIndex):
        self.listOfNodes = listOfNodes
        self.nextLayer = nextLayer
        self.prevLayer = None
        self.networkIndex = networkIndex
        
        self.outputs = None
        
    def setInputs(self, inputs):
        assert len(inputs) == len(self.listOfNodes)
        
        for i, inp in enumerate(inputs):
            self.listOfNodes[i].setOut(inp)
        
        self.outputs = inputs
        
class Network:
    def __init__(self, listOfLayers, costFunc, dCdX):
        self.listOfLayers = listOfLayers
        self.costFunc = costFunc
        self.dCdX = dCdX
        
    def feedForward(self, inputs):
        self.listOfLayers[0].setInputs(inputs)
        
        for layer in self.listOfLayers[1:]:
            layer.feedForward()
            
        return self.listOfLayers[-1].outputs
        
    def gradientDescent(self, trainingData, epochs, eta, testData=None):
        if testData:
            nTest = len(testData)
            
        n = len(trainingData)
        
        for j in range(epochs):
            trainingExample = random.choice(trainingData)
            
            self.backPropagate(trainingExample, eta)
            
            if testData:
                print "Epoch {0}: cost is {1}".format(j, self.evaluate(testData))
                
            else:
                print "Epoch {0} complete".format(j)
                
    def batchGradientDescent(self, batch, epochs, eta, testData=None):
        if testData:
            nTest = len(testData)
            
        n = len(batch)
        
        for j in range(epochs):
            self.batchBackPropagate(batch, eta)
            
            if testData:
                print "Epoch {0}: cost is {1}".format(j, self.evaluate(testData))
                
            else:
                print "Epoch {0} complete".format(j)
            
    def evaluate(self, testData):
        # figure out the sum of the costs of the test data points
        return sum([self.costFunc(self.feedForward(dataPoint[0]), dataPoint[1]) for dataPoint in testData])
        
    def feedForwardStartingAtLayer(self, startLayerIndex):
        for layer in self.listOfLayers[startLayerIndex:]:
            layer.feedForward()
            
        print self.listOfLayers[-1].outputs
        return self.listOfLayers[-1].outputs
        
    def evaluateStartingAtLayer(self, startLayerIndex, solution):
        return self.costFunc(self.feedForwardStartingAtLayer(startLayerIndex), solution)
    
    def backPropagate(self, dataPoint, eta):
        inputs = dataPoint[0]
        solutions = dataPoint[1]
        
        self.feedForward(inputs)
        
        for i in range(len(self.listOfLayers) - 1, 0, -1):
            layer = self.listOfLayers[i]
            layer.backPropagate(solutions, self.dCdX)
            
        before = self.costFunc(self.feedForward(dataPoint[0]), dataPoint[1])
        print "before", before
        
        for i in range(len(self.listOfLayers) - 1, 0, -1):
            layer = self.listOfLayers[i]
            layer.changeWeights(eta)

        after = self.costFunc(self.feedForward(dataPoint[0]), dataPoint[1])
        print "after", after
        
        assert before > after

    def batchBackPropagate(self, batch, eta):
        for dataPoint in batch:
            inputs = dataPoint[0]
            solutions = dataPoint[1]
            
            self.feedForward(inputs)
            
            for i in range(len(self.listOfLayers) - 1, 0, -1):
                layer = self.listOfLayers[i]
                layer.backPropagate(solutions, self.dCdX)
                
#        before = self.costFunc(self.feedForward(dataPoint[0]), dataPoint[1])
#        print "before", before
        
        for i in range(len(self.listOfLayers) - 1, 0, -1):
            layer = self.listOfLayers[i]
            layer.changeWeights(eta)

#        after = self.costFunc(self.feedForward(dataPoint[0]), dataPoint[1])
#        print "after", after
        
#        assert before > after        
            
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
    
    return [littleList + [0.1] for littleList in allSizeXMinus1Lists] + \
        [littleList + [0.9] for littleList in allSizeXMinus1Lists]
        
def crossEntropyCost(outputs, solutions):
    numOutputs = len(solutions)
    
    overallSum = 0
    
    for i in range(numOutputs):
        y = solutions[i]
        a = outputs[i]
        
        overallSum -= y * math.log(a) + (1 - y) * math.log(1 - a)
            
    overallSum /= numOutputs
    
    return overallSum
    
def crossEntropyCostPrime(outputs, solutions, outputIndex):
    numOutputs = len(solutions)
        
    return -1 * (solutions[outputIndex] / outputs[outputIndex] - \
         (1 - solutions[outputIndex]) / (1 - outputs[outputIndex])) / numOutputs
        
#trainingList = []

#for size4List in allSizeXLists(4):        
#    trainingExample = (size4List, [0.8*(sum(size4List) < 1.3)+0.1, #0.8*(sum(size4List) == 2.0)+0.1, 
 #                       0.8*(sum(size4List) == 2.8)+0.1, 0.8*(sum(size4List) == #3.6)+0.1])
    
#    trainingList += [trainingExample] * 10
    
trainingList = []
templateList = []

allSizeXLists2 = allSizeXLists(2)

#allSizeXList2 = [[0.1, 0.9]]

for size2List in allSizeXLists2:
    if ((size2List[0] == 0.9) ^ (size2List[1] == 0.9)):
        trainingExample = [size2List, [0.9, 0.1]]
    else:
        trainingExample = [size2List, [0.1, 0.9]]
    
    trainingList += [trainingExample] * 10
    templateList += [trainingExample]
                                
nodesPerLayer = int(sys.argv[1])
numLayers = int(sys.argv[2])    
        
inputLayer = InputLayer([InputNode(i) for i in range(nodesPerLayer)], None, 0)
listOfLayers = [inputLayer]

for i in range(numLayers - 1):
    listOfNodesInLayer = [Node([1./nodesPerLayer + random.random() * 0.1 for k in range(nodesPerLayer)], softNand, \
            dSoftNanddX, dSoftNanddW, j) for j in range(nodesPerLayer)]
    layer = Layer(listOfNodesInLayer, listOfLayers[-1], None, i+1)
    listOfLayers.append(layer)
    
for i in range(numLayers - 1):
    listOfLayers[i].nextLayer = listOfLayers[i+1]    
    
network = Network(listOfLayers, crossEntropyCost, crossEntropyCostPrime)    

#print network.feedForward([0.1, 0.9, 0.9, 0.1])
#print trainingList


first = network.feedForward([0.1, 0.1])
second = network.feedForward([0.1, 0.9])
third = network.feedForward([0.9, 0.1])
fourth = network.feedForward([0.9, 0.9])

network.batchGradientDescent(templateList, int(sys.argv[3]), 0.001)
#network.gradientDescent([[[0.1, 0.9], [0.9, 0.1]]], 1, 0.001)

#print network.listOfLayers[1].listOfNodes[0].weights, "hi"
#print network.listOfLayers[2].listOfNodes[0].weights
#print network.listOfLayers[3].listOfNodes[0].weights


def otherCost(a, y):
    return y[0] * math.log(a[0]) + (1-y[0]) * math.log(1-a[0])
    
def otherCostPrime(a, y, x=None):
    return y[0] / a[0] - (1-y[0]) / (1-a[0])


#print network.feedForward([0.1, 0.9, 0.9, 0.1])

print first, [0.1, 0.9], crossEntropyCost(first, [0.1, 0.9])
print second, [0.9, 0.1], crossEntropyCost(second, [0.9, 0.1])
print third, [0.9, 0.1], crossEntropyCost(third, [0.9, 0.1])
print fourth, [0.1, 0.9], crossEntropyCost(fourth, [0.1, 0.9])

print ''

print network.feedForward([0.1, 0.1]), [0.1, 0.9], crossEntropyCost(network.feedForward([0.1, 0.1]), [0.1, 0.9])
print network.feedForward([0.1, 0.9]), [0.9, 0.1], crossEntropyCost(network.feedForward([0.1, 0.1]), [0.9, 0.1])
print network.feedForward([0.9, 0.1]), [0.9, 0.1], crossEntropyCost(network.feedForward([0.1, 0.1]), [0.9, 0.1])
print network.feedForward([0.9, 0.9]), [0.1, 0.9], crossEntropyCost(network.feedForward([0.1, 0.1]), [0.1, 0.9])

#print trainingList

#print crossEntropyCost([0.20001], [0.3])
#print crossEntropyCost([0.2], [0.3])
#print crossEntropyCostPrime([0.2], [0.3], 0)

#print (crossEntropyCost([0.20001], [0.3]) - crossEntropyCost([0.2], [0.3])) * 100000
#print crossEntropyCostPrime([0.2], [0.3], 0)

#print (otherCost([0.20001], [0.3]) - otherCost([0.2], [0.3])) * 100000
#print otherCostPrime([0.2], [0.3], 0)
    
def printNetworkdCdOuts():
    for layer in network.listOfLayers:
        for node in layer.listOfNodes:
            print layer.networkIndex, node.layerIndex, node.dCdOut   
    
def checkNodedCdW(inputs, node, network):
    out1 = network.evaluate(inputs)
    
    node.weights[0] += 0.000001
    
    out2 = network.evaluate(inputs)
    
    print (out1 - out2) * 1000000
    
    print node.dCdW[0]
    
def checkNodedCdX(inputs, layer, nodeIndex, network):
    network.feedForward(inputs[0][0])
    
    out1 = network.evaluateStartingAtLayer(3, inputs[0][1])
    print "out1", out1
    
    layer.outputs[nodeIndex] += 0.000001
    
    out2 = network.evaluateStartingAtLayer(3, inputs[0][1])
    print "out2", out2
    
    node = layer.listOfNodes[nodeIndex]
    
    print (out1 - out2) * -1000000
    print node.dCdOut
        
 #   node.weights[0] += 0.000001
    
  #  out2 = network.evaluate(inputs)
    
   # print (out1 - out2) * 1000000
    
    #print node.dCdW[0]    

#checkNodedCdW([[[0.1, 0.9], [0.9, 0.1]]], network.listOfLayers[1].listOfNodes[0], network)
#checkNodedCdX([[[0.1, 0.9], [0.9, 0.1]]], network.listOfLayers[2], 0, network)