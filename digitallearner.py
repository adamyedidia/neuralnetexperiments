import random

class Gate:
    def __init__(self):
        self.binaryFunc = {(0, 0): random.randint(0, 1), (0, 1): random.randint(0, 1), (1, 0): random.randint(0, 1), (1, 1): random.randint(0, 1)}

class Network:
    def __init__(self, numLayers, gatesPerLayer, idMap):
        self.layerDict = {}
        for i in range(numLayers):
            self.layerDict[i] = {}

        for i in range(numLayers):
            for j in range(gatesPerLayer):
                self.layerDict[i][j] = Gate()

        self.idMap = idMap

    def evaluate(self, inputs):
        for i in range(len(self.layerDict)):
            layer = self.layerDict[i]
            
            newInputs = []
            for j in range(len(layer)):
                gate = layer[j]
  #              print self.idMap(i, j)[0], self.idMap(i, j)[1], inputs

                newInputs.append(gate.binaryFunc[(inputs[self.idMap(i, j)[0]], inputs[self.idMap(i, j)[1]])])

            inputs = newInputs
            # goddamn
            # all this is saying is "apply each gate to the pair of inputs from the previous layer, according to what the idMap
            # function tells you to do

        return inputs
    
    def trainGate(self, trainingExample, gate, errorFunc):
        bestToggle = None
        
        currentResult = self.evaluate(trainingExample[0])
        bestNumber = errorFunc(currentResult, trainingExample[1])
        
        for inputTuple in gate.binaryFunc:
            gate.binaryFunc[inputTuple] = 1 - gate.binaryFunc[inputTuple]
            
            result = self.evaluate(trainingExample[0])
            error = errorFunc(result, trainingExample[1])
            
            if error < bestNumber:
                bestNumber = error
                bestToggle = inputTuple
                
    def trainNetwork(self, layerDistribution, trainingExamples, numEpochs, errorFunc):
        
        
        for epochNum in range(1, numEpochs+1):
            print "Epoch", epochNum
            
            for example in trainingExamples:
                
                assert sum(layerDistribution) == 1
                assert len(layerDistribution) == len(self.layerDict)
            
                randomValue = random.random()
                layerIndex = 0
        
                while randomValue > layerDistribution[layerIndex]:
                     randomValue -= layerDistribution[layerIndex]
                     layerIndex += 1
             
                currentLayer = self.layerDict[layerIndex]
                randomGate = currentLayer[random.randint(0, len(currentLayer)-1)]

                self.trainGate(example, randomGate, errorFunc)

def listDistance(list1, list2):
    assert len(list1) == len(list2)
    return sum([list1[i] != list2[i] for i in range(len(list1))])
        



def areThereTwoOnesInARow():
    return [([0, 0, 0, 0], [0, 0, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 0, 0]),
        ([0, 0, 1, 0], [0, 0, 0, 0]),
        ([0, 0, 1, 1], [1, 1, 1, 1]),
        ([0, 1, 0, 0], [0, 0, 0, 0]),
        ([0, 1, 0, 1], [0, 0, 0, 0]),
        ([0, 1, 1, 0], [1, 1, 1, 1]),
        ([0, 1, 1, 1], [1, 1, 1, 1]),
        ([1, 0, 0, 0], [0, 0, 0, 0]),
        ([1, 0, 0, 1], [1, 1, 1, 1]),
        ([1, 0, 1, 0], [0, 0, 0, 0]),
        ([1, 0, 1, 1], [1, 1, 1, 1]),
        ([1, 1, 0, 0], [1, 1, 1, 1]),
        ([1, 1, 0, 1], [1, 1, 1, 1]),
        ([1, 1, 1, 0], [1, 1, 1, 1]),
        ([1, 1, 1, 1], [1, 1, 1, 1]),
        ]

def allInputs():
    return [[0, 0, 0, 0],
    [0, 0, 0, 1], 
    [0, 0, 1, 0],
    [0, 0, 1, 1],  
    [0, 1, 0, 0], 
    [0, 1, 0, 1], 
    [0, 1, 1, 0], 
    [0, 1, 1, 1], 
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1], 
    [1, 1, 0, 0], 
    [1, 1, 0, 1], 
    [1, 1, 1, 0],
    [1, 1, 1, 1],
    ]

g = Gate()
n = Network(4, 4, (lambda x, y: (y, (y+1)%4)))
n.trainNetwork([0.6, 0.25, 0.1, 0.05], areThereTwoOnesInARow(), 2000, listDistance)

for x in allInputs():
    print x, n.evaluate(x)

