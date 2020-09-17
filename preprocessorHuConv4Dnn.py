import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from utils import getHuMoments

class PreprocessorHuConv4Dnn():
    
    def __init__(self, cnn, removeHuPreprocess=False, huTransformOutFunc="default"):
        self.cnnPrefix = self.getCnnPrefixAsSequential(cnn)
        self.removeHuPreprocess = removeHuPreprocess
        self.huTransformOutFunc = huTransformOutFunc
        
    def preprocessData(self, data):
        (x_train, y_train),(x_test, y_test) = data 
        if self.removeHuPreprocess:
            x_train = self.preprocessWithConv(x_train, self.cnnPrefix)
            x_test = self.preprocessWithConv(x_test, self.cnnPrefix)
        else:
            x_train = self.__preprocessData(x_train, self.cnnPrefix, self.huTransformOutFunc)
            x_test = self.__preprocessData(x_test, self.cnnPrefix, self.huTransformOutFunc)
        return (x_train, y_train),(x_test, y_test)
        
    @staticmethod
    def preprocessHuMoments(xData, huTransformOutFunc="default"):
        print("   ----> Preprocessing Data to HuMoments")
        getHuMoms = lambda x: getHuMoments(x, huTransformOutFunc)
        xData = list(map(getHuMoms, xData))
        xData = np.asarray(xData, dtype=np.float32)
        print("   ----> Preprocessing Data to HuMoments DONE")
        return xData
        
    @staticmethod
    def preprocessWithConv(xData, cnnPrefix):
        print("   ----> Preprocessing Data with Conv Layers")
        xData = cnnPrefix.predict(xData)
        print("   ----> Preprocessing Data with Conv Layers Done")
        return xData
        
    @classmethod
    def getCnnPrefixAsSequential(cls, cnn): 
        convLayers = cls.__getConvLayers( cnn.model )
        convFeaturePreprocess = Sequential()
        [convFeaturePreprocess.add(l) for l in convLayers]
        return convFeaturePreprocess    
    
    @classmethod
    def __preprocessData(cls, xData, cnnPrefix, huTransformOutFunc):
        p1 = cls.preprocessWithConv(xData, cnnPrefix)
        p2 = cls.preprocessHuMoments(xData, huTransformOutFunc)
        xData = np.concatenate((p1, p2), axis=1)
        return xData
        
    @classmethod
    def __getConvLayers(cls, model):
        return model.layers[:cls.__get1stDenseLayerIndex(model)]
        
    @staticmethod
    def __get1stDenseLayerIndex(model):
        for i in range(len(model.layers)):
            if type(model.get_layer(index=i)) == Dense:
                return i