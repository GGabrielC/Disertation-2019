
class ANN():
    def __init__(self, builder, preprocessor=None):
        self.model = builder.build()
        self.builder = builder
        self.preprocessor = preprocessor
        
    def train(self, data, epochs, accuracy4Output=True, verbose=False):
        if self.preprocessor != None:
            data = self.preprocessor.preprocessData(data)
        print("   ----> Training {0} Model (epochs: {1})".format(self.getModelType(), epochs))
        trainData, testData = data
        self.model.fit(*trainData, epochs=epochs, verbose=verbose)
        print("   ----> Training {0} Model DONE (epochs: {1})".format(self.getModelType(), epochs))
        if accuracy4Output: 
            return  self.evaluate(testData, verbose=verbose)[1]
        
    def evaluate(self, testData, verbose=False):
        modelType = self.getModelType()
        print("\n   ----> Evaluating {0}".format(modelType))
        outEvaluate = self.model.evaluate(*testData, verbose=verbose)
        print(self.formatTestResults(outEvaluate[1]))
        print("   ----> Evaluating {0} DONE".format(modelType))
        return outEvaluate
        
    def getModelType(self):
        return self.builder.modelType
    
    def basicCheck(self, data, epochs=1, verbose=True, isDetailed=True):
        accuracy = self.train(data, epochs=epochs, verbose=verbose)
        self.summary(isDetailed)
        print(self.formatTestResults(accuracy))
    
    def summary(self, isDetailed=True):
        self.printModelSummary(isDetailed)
        
    def printModelSummary(self, isDetailed=True):
        modelType = self.builder.modelType
        print("\n   ---> {0} SUMMARY".format(modelType))
        print("input_shape = {0}".format(self.model.input_shape))
        print("output_shape = {0}".format(self.model.output_shape))
        if isDetailed == True:
            self.model.summary()
        print("   ---> {0} END SUMMARY\n".format(modelType))

    def formatTestResults(self, accuracy):
        return "\n   ---> Test Results {0}: accuracy: {1}\n".format(self.builder.modelType, accuracy)