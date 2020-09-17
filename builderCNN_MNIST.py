import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, Reshape

class BuilderCNN_MNIST():
    
    def __init__(self, convOutputSize=16, removeConvLayers=False):
        self.convOutputSize = convOutputSize
        self.removeConvLayers = removeConvLayers
        self.modelType = "CNN"
        if self.removeConvLayers: 
            self.modelType = "DNN"
        
    def build(self):
        model = Sequential()
        if not self.removeConvLayers:
            self.addConvLayers(model, self.convOutputSize)
        self.addDenseLayers(model)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model
    
    @staticmethod
    def addConvLayers(model, convOutputSize):
        #28-2 26-2 24/2 12-2 10-2 8/2 4-2 2/2 1
        lastConvChannels = convOutputSize
        
        model.add(Reshape((28,28,1), input_shape=(28,28)))
        model.add(Lambda(lambda x: x/255.0))
        
        model.add(Conv2D( 30, (3, 3), activation='relu', input_shape = (28,28,1) ))
        model.add(Conv2D( 25, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D( 25, (3, 3), activation='relu', input_shape = (28,28,1) ))
        model.add(Conv2D( 20, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(lastConvChannels, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())

    @staticmethod
    def addDenseLayers(model):
        model.add(Dense(90, activation='relu'))
        model.add(Dense(70, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(10, activation='softmax'))