import sys
from preprocessorHuConv4Dnn import PreprocessorHuConv4Dnn
from builderCNN_MNIST import BuilderCNN_MNIST
from ann import ANN
from colorama import init
from colorama import Back
init()

class PerformanceComparator():
    def __init__(self, verboseTrainEvalProgress=False):
        self.indexExpectedWinner = 1
        self.indexUnwantedWinners = [2]
        self.verbose = verboseTrainEvalProgress
        
    def compareAnnAccuracies(self, data, epochsList):
        [print("--------> compareAnnAccuracies START") for i in range(10)]
        print("--------> epochsList: {0}".format(epochsList))
        print("--------> sum epochsList: {0}".format(sum(epochsList)))
        return self.__iteratedCompare(data, epochsList)

    def __iteratedCompare(self, data, epochsList):
        history = []
        history.append(["epochs", ["cnn1", "dnn_huConv", "dnn_conv", "cnn2", "dnn_conv2"], ["descSortedIndexes"]])
        
        try:
            for epochs in epochsList:
                print("--------> Starting Iteration {0} with epochs:{1}".format(len(history), epochs))
                self.printResults(history, self.verbose, finished=False); sys.stdout.flush()
                
                anns = self.getAnns()
                results = [ann.train(data, epochs, verbose=self.verbose) for ann in anns]
                sortedIndexes = self.getSortedIndexes(results)
                history.append([epochs, results, sortedIndexes])
                
                if len(history) == 2:
                    [ann.summary() for ann in anns]
                
        except Exception as e:
            print("   -----> ERROR Something went wrong <-------")
            print(e)
        finally: 
            self.printResults(history, self.verbose)
        return history
    
    def __getWinner(self, results):
        index = results.index(max(results))
        if self.indexExpectedWinner == None:
            return index
        if index == self.indexExpectedWinner:
            index = str(index)+" VICTORY"
        elif index in self.indexUnwantedWinners:
            index = str(index)+" OOPS"
        return str(index)
        
    @classmethod
    def printResults(cls, history, prettyPrint=False, finished=True):
        if not len(history)>1: 
            return
        if finished:
            print("------>>>   RESULTS:")
        else: 
            print("------>>>   RESULTS so far:")
        if prettyPrint:
            [cls.prettyPrintResultLine(i) for i in history]
        else: [print(i) for i in history]
        sys.stdout.flush()
        
    @staticmethod
    def getAnns():
        builderNoConv = BuilderCNN_MNIST(removeConvLayers=True)
        
        cnn1 = ANN(BuilderCNN_MNIST(18))
        dnn_huConv = ANN(builderNoConv, PreprocessorHuConv4Dnn(cnn1))
        dnn_conv = ANN(builderNoConv, PreprocessorHuConv4Dnn(cnn1, removeHuPreprocess=True))
        
        cnn2 = ANN(BuilderCNN_MNIST(18+7))
        dnn_conv2 = ANN(builderNoConv, PreprocessorHuConv4Dnn(cnn2, removeHuPreprocess=True))
        
        return [cnn1, dnn_huConv, dnn_conv, cnn2, dnn_conv2]
        
    @staticmethod
    def getSortedIndexes(scores):
        l = list(zip(range(len(scores)), scores))
        l.sort(key=lambda x:x[1])
        l = [i[0] for i in l]
        l.reverse()
        return l
        
    @staticmethod
    def prettyPrintResultLine(line):
        colors = [Back.BLUE, Back.GREEN, Back.CYAN, Back.MAGENTA, Back.RED, Back.RESET]
        
        Back.RESET
        strLine = "[" + str(line[0]) + ",  ["
        for i,v in zip(range(len(line[1])), line[1]):
            strLine += colors[i] + str(v) + Back.RESET+", "
        strLine += "],  ["
        for i in line[2]:
            color = i
            if i == "descSortedIndexes":
                color = 5
            strLine += colors[color] + str(i) + Back.RESET+", "
        strLine += "] ]"
        print(strLine)
    
        
        
        
        