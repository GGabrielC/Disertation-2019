import numpy as np
import sys
import os 
import datetime

from preprocessorHuConv4Dnn import PreprocessorHuConv4Dnn
from builderCNN_MNIST import BuilderCNN_MNIST
from ann import ANN
from stats import doStatistics

from utils import getMnistData
from utils import logHu, softsign, softLog, mnistMaxLog
from performanceComparator import PerformanceComparator
    # Python 3.7.2
    
def main(verbose, list):
    epochsList = list
    data = getMnistData()
    results = PerformanceComparator(verbose).compareAnnAccuracies(data, epochsList)
    doStatistics(results)

######################################
######################################        

def setStdOut(fileName="stdout.txt"):
    filePath = getCurrDirPath()+"\\"+fileName
    original = sys.stdout
    print("\n   {0}\n   !!! stdout changed to {1} !!!\n   {0}\n".format("!"*40,filePath))
    sys.stdout = open(filePath, 'a+')
    
    [print("#"*60) for i in range(4)] 
    print("DATE: "+str(datetime.datetime.now()))
    [print("#"*60) for i in range(4)]
    
    return (original, filePath)

def getCurrDirPath():
    return os.path.dirname(os.path.realpath(__file__))

    # Python 3.7.2
    
if __name__== "__main__":
    if len(sys.argv) < 2:
        changeStdout = False
        epochslist = [1]
    else:
        changeStdout = sys.argv[1]=="1" # False True
        epochslist = list(map(lambda x: int(x), sys.argv[2:]))
    print("changeStdout: {0}\nepochsList: {1}".format(changeStdout, epochslist))
    
    stdoutChanged = False
    if changeStdout:
        oldStdout, newStdoutPath = setStdOut(); 
        stdoutChanged = True
    verbose = not stdoutChanged
    
    main(verbose, epochslist)
    
    if stdoutChanged:
        file, sys.stdout = sys.stdout, oldStdout; file.close()
    