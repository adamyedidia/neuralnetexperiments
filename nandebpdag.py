import random

# These are the values that are passed into the nands; they represent the probability that the true value is 1

# These are more or less just mutable Floats 
class Value:
    def __init__(self):
        self.value = None
        
    def set(self, value):
        self.value = value

class Nand:
    def __init__(self, weights, ID, inputList, output):
        assert len(weights) == len(inputList)
        
        # UNCHANGING
        self.ID = ID
        self.inputList = inputList
        self.output = output
        
        # MUTABLE
        self.weights = weights
    
    def getInputs(self):
        return [inp.value for inp in self.inputList] 
        
    def setInputs(self, newInputs):
#        print newInputs, self.inputList
        assert len(newInputs) == len(self.inputList)
        
        for i, value in enumerate(newInputs):
            self.inputList[i].set(value)
        
    def feedForward(self):
        inputs = self.getInputs()
        
        numInputs = len(inputs)
        
        assert numInputs == len(self.weights)
        
        isThereAnActive0Product = 1.0
        
        for i in range(numInputs):
            x = inputs[i]
            w = self.weights[i]
            
            # Multiply by the probability that either it's inactive or it's not a 0
            isThereAnActive0Product *= x + 1 - w - x*(1-w)
            
        self.output.set(1.0 - isThereAnActive0Product)
     
    def updateFromOutput(self):
                
        currentWeights = self.weights
        
        inputs = self.getInputs()
        
#        weightContributionFrom0 = numTimesList(1-self.output.value, updateWeightsFrom0(inputs, currentWeights))
#        weightContributionFrom1 = numTimesList(self.output.value, updateWeightsFrom1(inputs, currentWeights))
        
#        print "before weights", self.weights
#        print "before inputs", inputs
#        print "output", self.output.value
        
        if self.output.value < 1e-10:
            self.weights = updateWeightsFrom0(inputs, self.weights)
            
        elif self.output.value > 1.0 - 1e-10:
            self.weights = updateWeightsFrom1(inputs, self.weights)
            
        else:
            weightContributionFrom0 = numTimesList(1-self.output.value, updateWeightsFrom0(inputs, self.weights))
            weightContributionFrom1 = numTimesList(self.output.value, updateWeightsFrom1(inputs, self.weights))
        
            self.weights = listPlusList(weightContributionFrom0, weightContributionFrom1)
                
#        inputContributionFrom0 = numTimesList(1-self.output.value, updateInputsFrom0(inputs, currentWeights))
#        inputContributionFrom1 = numTimesList(self.output.value, updateInputsFrom1(inputs, currentWeights))

        if self.output.value < 1e-10:
            self.setInputs(updateInputsFrom0(inputs, self.weights))
            
        elif self.output.value > 1.0 - 1e-10:
            self.setInputs(updateInputsFrom1(inputs, self.weights))
            
        else:
            inputContributionFrom0 = numTimesList(1-self.output.value, updateInputsFrom0(inputs, self.weights))
            inputContributionFrom1 = numTimesList(self.output.value, updateInputsFrom1(inputs, self.weights))
        
            self.setInputs(listPlusList(inputContributionFrom0, inputContributionFrom1))
        
#        print "after weights", self.weights
#        print "after inputs", self.getInputs()
#        print ""
        
    def roundWeights(self):
        self.weights = [round(w) for w in self.weights] 
        
class DAG:
    def __init__(self, numInputs, numNodes):
        self.inputValues = [Value() for i in range(numInputs)]
        
        self.listOfNands = []
        
        runningListOfInputs = self.inputValues[:]
        
        for i in range(numNodes):
            newValue = Value()
            
            initialWeights = [1./numInputs]*len(runningListOfInputs)
            
            newNand = Nand(initialWeights, i, runningListOfInputs, newValue)
            
            runningListOfInputs = runningListOfInputs + [newValue]
            self.listOfNands.append(newNand)
            
        self.output = newValue    
            
    def setInputValues(self, newInputs):
        assert len(newInputs) == len(self.inputValues)
        
        for i, value in enumerate(newInputs):
            self.inputValues[i].set(value)        
            
    def feedForward(self, inputs):
        self.setInputValues(inputs)
        
        for nand in self.listOfNands:        
            nand.feedForward()
            
        return self.output.value
        
    def updateFromOutput(self, correctOutput):
        self.output.set(correctOutput)
                
        for i in range(len(self.listOfNands) - 1, -1, -1):
            nand = self.listOfNands[i]
            
            nand.updateFromOutput()    
                    
    def train(self, trainingSet, epochs=1):
        trainingSetCopy = trainingSet[:]
        
        for i in range(epochs):
            random.shuffle(trainingSetCopy)
            
            for dataPoint in trainingSetCopy:
                inputs = dataPoint[0]
                correctOutput = dataPoint[1]
                
                self.feedForward(inputs)
                self.updateFromOutput(correctOutput)
                
            print "Epoch", i, "complete."
    
    def roundWeights(self):
        for nand in self.listOfNands:
            nand.roundWeights()      
            
def numTimesList(x, l):
    return [x * i for i in l]

def listPlusList(l1, l2):
    lenL1 = len(l1)
    assert lenL1 == len(l2)

    return [l1[i] + l2[i] for i in range(lenL1)]

def allBooleanTuplesOfLength(x):
    if x == 0:
        return [()]
        
    allSizeXMinus1Tuples = allBooleanTuplesOfLength(x-1)
    
    return [littleTuple + tuple([True]) for littleTuple in allSizeXMinus1Tuples] + \
        [littleTuple + tuple([False]) for littleTuple in allSizeXMinus1Tuples]
        

# This NAND gate outputs a 0. We must update in response!

# ALL active inputs must be 1's. So if a weight is a 1, then the corresponding input MUST be 1!
#def updateInputsFrom0(inputs, weights):
#    assert len(inputs) == len(weights)
    
#    newInputs = []
    
#    for i in range(len(weights)):
#        x = inputs[i]
#        w = weights[i]
        
#        newInputs.append((1-w)*x + w)
        
#    return newInputs
 
def orProb(x, y):
     return x + y - x*y
 
def updateInputsFrom0(inputs, weights):
    assert len(inputs) == len(weights)
    
    newInputs = []
    
    for i in range(len(weights)):
        x = inputs[i]
        w = weights[i]
        
        newInputs.append(x / (x + (1-x)*(1-w)))
        
    return newInputs 
    
# This NAND gate output a 1. We must update in response!

# Some active input must be a 0. 
def updateInputsFrom1Slower(inputs, weights):
    numWeights = len(weights)
    
    assert len(inputs) == len(weights)
    
    listOfOnnessAssignments = allBooleanTuplesOfLength(len(inputs))
    
    onnessDict = {}
    
    sumAllPosteriors = 0.0
    
    for booleanTuple in listOfOnnessAssignments:
        
        # Multiplies together the probabilities that the inputs are this onness level in the first place
        priorProduct = 1.0
        
        # Multiplies together the probabilities that each input that's a 0 is inactive (one of them must be active, so we'll do 1-this at the end)
        likelihoodProduct = 1.0
        
        for i, onness in enumerate(booleanTuple):
            if onness:
                priorProduct *= inputs[i]
            else:
                priorProduct *= 1.0 - inputs[i]
                
            if not onness:
                likelihoodProduct *= 1.0 - weights[i]
                
        likelihoodProduct = 1.0 - likelihoodProduct
        
        onnessDict[booleanTuple] = priorProduct * likelihoodProduct
        
        sumAllPosteriors += onnessDict[booleanTuple]
        
    # Now we need to figure out the probability that each input is on or not
    newInputs = []
    
    for i in range(numWeights):
        
        tuplesBefore = allBooleanTuplesOfLength(i)
        tuplesAfter = allBooleanTuplesOfLength(numWeights - i - 1)
        
        truthLikelihood = 0.0
        
        for beforeTuple in tuplesBefore:
            for afterTuple in tuplesAfter:
                trueTuple = beforeTuple + tuple([True]) + afterTuple
                
#                print trueTuple, onnessDict[trueTuple]
                
                truthLikelihood += onnessDict[trueTuple]
        
#        print truthLikelihood, sumAllPosteriors
        
        # Weird 0/0 case, just take the prior
#        if truthLikelihood == 0 and sumAllPosteriors == 0:
#            newInputs.append(inputs[i])
        
#        else:
        newInputs.append(truthLikelihood / sumAllPosteriors)
        
    return newInputs
    
def updateInputsFrom1(inputs, weights):
    numWeights = len(weights)
    
    assert len(inputs) == numWeights
        
    isThereAnActive0Product = 1
    
    for i in range(numWeights):
        x = inputs[i]
        w = weights[i]
        
        isThereAnActive0Product *= orProb(x, 1-w)
        
    newInputs = []
    
    for i in range(numWeights):
        x = inputs[i]
        w = weights[i]
        
        if orProb(x, 1-w) == 0.0:
            newInputs.append(0.0)
        else:
            oddsOfEverythingElseAlright = 1 - isThereAnActive0Product/(orProb(x, 1-w))
      
            everythingAlrightDespiteMe = x * oddsOfEverythingElseAlright
            everythingAlrightWithoutMe = (1-x) * orProb(w, oddsOfEverythingElseAlright)
        
            newInputs.append(everythingAlrightDespiteMe / (everythingAlrightDespiteMe + everythingAlrightWithoutMe))
    
#    print newInputs
    
    return newInputs
# This NAND gate outputs a 0. We must update in response!

# ALL active inputs must be 1's. So if an input is a 0, then the corresponding weight MUST be 0!
def updateWeightsFrom0(inputs, weights):
    assert len(inputs) == len(weights)
    
    newWeights = []
    
    for i in range(len(weights)):
        x = inputs[i]
        w = weights[i]
        
        # x is prob input is a 1 (that's the okay probability)
        # In this case we don't know what the weight should be so we use its prior odds
        newWeights.append(w*x / (1-w + w*x))
        
    return newWeights

# The NAND gate outputs a 1. We must update in response!

# Some active input must be a 0.
def updateWeightsFrom1(inputs, weights):
    numWeights = len(weights)
    
    assert len(inputs) == numWeights
    
    isThereAnActive0Product = 1
    
    for i in range(numWeights):
        x = inputs[i]
        w = weights[i]
        
        isThereAnActive0Product *= orProb(x, 1-w)
    
    newWeights = []
    
    for i in range(numWeights):
        x = inputs[i]
        w = weights[i]
        
        if orProb(x, 1-w) == 0.0:
            newWeights.append(1.0)
        # Think about it! If the or is 0, then w has to be 1, there's no two ways about it
        else:    
            oddsOfEverythingElseAlright = 1 - isThereAnActive0Product/(orProb(x, 1-w))
        
            everythingAlrightThanksToMe = w * orProb(1-x, oddsOfEverythingElseAlright)
            everythingAlrightWithoutMe = (1-w) * oddsOfEverythingElseAlright
        
            newWeights.append(everythingAlrightThanksToMe / (everythingAlrightThanksToMe + everythingAlrightWithoutMe))

    return newWeights
# The NAND gates outputs a 1. We must update in response!

# Some active input must be a 0.         
def updateWeightsFrom1Slower(inputs, weights):
    numWeights = len(weights)
    
    assert len(inputs) == len(weights)
    
    listOfActivityAssignments = allBooleanTuplesOfLength(len(inputs))
    
    activityDict = {}
    
    # The sum of all posterior probabilities (used for renormalizing)
    sumAllPosteriors = 0.0
    
    for booleanTuple in listOfActivityAssignments:
        
        # Multiplies together the probabilities that the weights are this activity level in the first place
        priorProduct = 1.0
        
        # Multiplies together the probabilities that each active input is a 1 (one of them must be a 0, so we'll do 1-this at the end)
        likelihoodProduct = 1.0
        
        for i, activity in enumerate(booleanTuple):
            if activity:
                priorProduct *= weights[i]
            else:
                priorProduct *= 1.0 - weights[i]
            
            if activity:
                likelihoodProduct *= inputs[i]
            
        likelihoodProduct = 1.0 - likelihoodProduct
        
#        print priorProduct, likelihoodProduct
        
        activityDict[booleanTuple] = priorProduct * likelihoodProduct
                
        sumAllPosteriors += activityDict[booleanTuple]                 
                        
    # Now we need to figure out the probability that each input is active or not
    newWeights = []
    
    for i in range(numWeights):
        
        tuplesBefore = allBooleanTuplesOfLength(i)
        tuplesAfter = allBooleanTuplesOfLength(numWeights - i - 1)
        
        truthLikelihood = 0.0
        
        for beforeTuple in tuplesBefore:
            for afterTuple in tuplesAfter:
                trueTuple = beforeTuple + tuple([True]) + afterTuple
                
#                print trueTuple, activityDict[trueTuple]
                
                truthLikelihood += activityDict[trueTuple]
                
#        print truthLikelihood, sumAllPosteriors        
                
#        if truthLikelihood == 0 and sumAllPosteriors == 0:
#            newWeights.append(weights[i])        
        
#        else:        
        newWeights.append(truthLikelihood / sumAllPosteriors)
        
    return newWeights

def allSizeXLists(x):
    if x == 0:
        return [[]]
        
    allSizeXMinus1Lists = allSizeXLists(x - 1)
    
    return [littleList + [0.0] for littleList in allSizeXMinus1Lists] + \
        [littleList + [1.0] for littleList in allSizeXMinus1Lists]

print updateInputsFrom0([0, 0.1], [0.2, 0.4]), updateWeightsFrom0([0, 0.1], [0.2, 0.4])        

trainingSet = [[[0.0, 0.0], 0.0], [[0.0, 1.0], 1.0], [[1.0, 0.0], 1.0], [[1.0, 1.0], 0.0]]

dag = DAG(2, 4)

#print updateWeightsFrom1([0.2, 0.3], [0.1, 0.4])
#print updateWeightsFrom1Faster([0.2, 0.3], [0.1, 0.4])

#print updateInputsFrom1([0.2, 0.3], [0.1, 0.4])
#print updateInputsFrom1Faster([0.2, 0.3], [0.1, 0.4])

dag.train(trainingSet, 20)

print dag.feedForward([0.0, 0.0])                
print dag.feedForward([0.0, 1.0])                
print dag.feedForward([1.0, 0.0])                
print dag.feedForward([1.0, 1.0])                
                
trainingSet = []

allSize3Lists = allSizeXLists(3)

for littleList in allSize3Lists:
    if sum(littleList) > 1:
        trainingSet.append([littleList, 1.0])
    else:
        trainingSet.append([littleList, 0.0])

dag = DAG(3, 4)

dag.train(trainingSet, 10)
#dag.roundWeights()

for littleList in allSize3Lists:
    print littleList, dag.feedForward(littleList)

#print updateInputsFrom0([0.3, 0.01], [0.4, 0.99])
#print updateInputsFrom1([0.3, 0.01], [0.4, 0.99])                
#print updateWeightsFrom0([0.3, 0.01], [0.4, 0.99])
#print updateWeightsFrom1([0.3, 0.01], [0.4, 0.99])
#print ""
#print updateInputsFrom0([0.8, 0.7, 0.8, 0.6, 0.9], [0.3, 0.2, 0.2, 0.1, 0.5])
#print updateInputsFrom1([0.8, 0.7, 0.8, 0.6, 0.9], [0.3, 0.2, 0.2, 0.1, 0.5])
#print updateWeightsFrom0([0.8, 0.7, 0.8, 0.6, 0.9], [0.3, 0.2, 0.2, 0.1, 0.5])
#print updateWeightsFrom1([0.8, 0.7, 0.8, 0.6, 0.9], [0.3, 0.2, 0.2, 0.1, 0.5])
#print ""
#print updateInputsFrom0([0.99, 0.99], [0.99, 0.99])
#print updateInputsFrom1([0.99, 0.99], [0.99, 0.99])
#print updateWeightsFrom0([0.99, 0.99], [0.99, 0.99])
#print updateWeightsFrom1([0.99, 0.99], [0.99, 0.99])
