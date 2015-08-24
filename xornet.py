from pybrain.datasets import SupervisedDataSet
from pybrain.structure.modules import ReluLayer, SigmoidLayer
import string

class Node:
    def __init__(self, nodeType, layer, ID, amInputNode=False, amOutputNode=False):
        self.nodeType = nodeType
        self.layer = layer
        self.ID = ID 
        self.amInputNode = amInputNode
        self.amOutputNode = amOutputNode

    def toSpiceNetlist(self, weightDictionary):
        marker = str(self.layer) + "_" + str(self.ID)
        
        outString = "* Circuitry for Node " + marker + "\n\n"
        
        if self.amInputNode:                    
            outString += "I" + marker + " 0 " + marker + "a 0mA\n"
            outString += "V" + marker + " " + marker + "a " + marker + "b 0\n"
            outString += "R" + marker + " 0 " + marker + "b 1k\n\n"
                        
        else:       
            print self.layer, self.ID
            for otherNodeID in weightDictionary[self.layer][self.ID]:
                prevMarker = str(self.layer-1) + "_" + str(otherNodeID)
                fullMarker = str(self.layer) + "_" + str(otherNodeID) + "_" + str(self.ID)
            
                outString += "F" + fullMarker + " 0 " + marker + "a V" + prevMarker + " " + \
                     str(weightDictionary[self.layer][self.ID][otherNodeID]) + "\n"
            
            if self.nodeType == "ReluLayer":         
                outString += "R" + marker + " 0 " + marker + "a 1000k\n"
                
            outString += "V" + marker + " " + marker + "a " + marker + "b 0\n"
            
            if self.nodeType == "ReluLayer":
                outString += "D" + marker + " " + marker + "b " + marker + "c 1mA_diode\n"
            elif self.nodeType == "LinearLayer":
                outString += "V" + marker + "b " + marker + "b " + marker + "c 0\n"
            
            if self.amOutputNode:
                outString += "Rout " + marker + "c 0 1k\n\n"
            else:
                outString += "V" + marker + "c " + marker + "c 0 0\n\n"
                
        return outString
                 
def neuralNetToSpiceNetlist(weightDictionary, nodeDictionary):
    outString = ""
    
    for i in nodeDictionary:
        for j in nodeDictionary[i]:    
            outString += nodeDictionary[i][j].toSpiceNetlist(weightDictionary)
    
    outString += ".model 1mA_diode D (Is=100pA n=1.679)\n"
    
    output = open("/Users/adamyedidia/Documents/MacSpice/neuralnet.cir", "w")
    
    output.write(outString)




dataModel = [
    [(0,0), (0,)],
    [(0,1), (1,)],
    [(1,0), (1,)],
    [(1,1), (0,)],
]

ds = SupervisedDataSet(2, 1)
for input, target in dataModel:
    ds.addSample(input, target)

# create a large random data set
import random
random.seed()
trainingSet = SupervisedDataSet(2, 1);
for ri in range(0,1000):
    input,target = dataModel[random.getrandbits(2)];
    trainingSet.addSample(input, target)

from pybrain.tools.shortcuts import buildNetwork
net = buildNetwork(2, 2, 1, bias=False, hiddenclass=ReluLayer) #, outclass=ReluLayer)



from pybrain.supervised.trainers import BackpropTrainer
trainer = BackpropTrainer(net, ds, learningrate = 0.001, momentum = 0.99)
trainer.trainUntilConvergence(verbose=True,
                              trainingData=trainingSet,
                              validationData=ds,
                              maxEpochs=10)

def pesos_conexiones(n):
    weightDictionary = {}
    
    # maps from layers to the number of neurons they contain
    layerDictionary = {}

    layerTypeDictionary = {}

    nodeDictionary = {}

    for i, mod in enumerate(n.modulesSorted):
        weightDictionary[i] = {}
        layerDictionary[i] = 0
        layerTypeDictionary[i] = str(mod)[1:string.find(str(mod), " ")]
        nodeDictionary[i] = {}

        

    for i, mod in enumerate(n.modulesSorted):
        for conn in n.connections[mod]:
            for cc in range(len(conn.params)):
                weightDictionary[i+1][conn.whichBuffers(cc)[1]] = {}
                
            for cc in range(len(conn.params)):            
                weightDictionary[i+1][conn.whichBuffers(cc)[1]][conn.whichBuffers(cc)[0]] = conn.params[cc]
                layerDictionary[i+1] = max(layerDictionary[i+1], conn.whichBuffers(cc)[1]+1)

                layerDictionary[i] = max(layerDictionary[i], conn.whichBuffers(cc)[0]+1)    
    
    for i in layerDictionary:
        
        for j in range(layerDictionary[i]):
            nodeDictionary[i][j] = Node(layerTypeDictionary[i], i, j, (i==0), \
                (i==len(layerDictionary)-1))
                
    print layerDictionary
    return weightDictionary, nodeDictionary
                              

                            
#convertToSpice(net)

weightDictionary, nodeDictionary = pesos_conexiones(net)
print nodeDictionary
neuralNetToSpiceNetlist(weightDictionary, nodeDictionary)

print '0,0->', net.activate([0,0])
print '0,1->', net.activate([0,1])
print '1,0->', net.activate([1,0])
print '1,1->', net.activate([1,1])
