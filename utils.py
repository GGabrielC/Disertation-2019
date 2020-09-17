import cv2
import numpy as np
import tensorflow as tf
import os
import skimage
import math
from math import log10, copysign
from numpy import array
from random import shuffle
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Reshape
from keras.models import load_model
from random import randint
from collections import Counter
import sys
epsilon = sys.float_info.epsilon

belgiumTrainPath = "C:\\Users\\Gabriel\\Diz\\Proj\\Belgium\\BelgiumTSC_Training\\Training"
belgiumTestPath = "C:\\Users\\Gabriel\\Diz\\Proj\\Belgium\\BelgiumTSC_Testing\\Testing"
germanPath = "C:\\Users\\Gabriel\\Diz\\Proj\\German\\train_Images\\"

def isNestedList(l):
    return type(l[0])==list

def shuffleData(entries, labels):
    pairs = list(zip(entries,labels))
    shuffle(pairs)
    return tuple(zip(*pairs))
    
def load_data(data_dir=None):
    if data_dir==None:
        data_dir = belgiumTrainPath
        
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
            
    images, labels = shuffleData(images, labels)
    return images, labels

def matchShapes(img1, img2):
    im1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return cv2.matchShapes(im1, im2, cv2.CONTOURS_MATCH_I2,0)

def getLogBettyNumbers():
    for i in range(6):
        imgColor = cv2.imread("letters/{0}.png".format(i))
        imgGray = cv2.cvtColor(imgColor, cv2.COLOR_BGR2GRAY)
        _, imgBinary = cv2.threshold(imgGray, 128, 255, cv2.THRESH_BINARY)
        
        results = [float("{0:.4f}".format(i)) for i in getHuMoments(imgBinary, logHu)]
        print(i)
        [print("H[{0}] = {1}".format(j,results[j])) for j in range(len(results))]

        
        cv2.imshow("image{0}".format(i), imgBinary)
    cv2.waitKey(60000)

def getHuMoments(img, transformOutFunc="default", thresholdFunc="default"):
    if transformOutFunc == "default":
        transformOutFunc = softLog # logHu softsign softLog mnistMaxLog
    
    huMoments = getRawHuMoments(img, thresholdFunc)
    huMoments = map(transformOutFunc,huMoments) if transformOutFunc else huMoments

    return list(huMoments)

def getRawHuMoments(img, thresholdFunc="default"):
    if thresholdFunc == "default":
        thresholdFunc = thresholdBinary
    
    img = asGray(img)
    img = thresholdBinary(img) if thresholdFunc else img
    huMoments = cv2.HuMoments(cv2.moments(img)).tolist()
    huMoments = np.array([i[0] for i in huMoments])
    return huMoments
    
def thresholdBinary(img, a=128, b=255):
    return cv2.threshold(img, a, b, cv2.THRESH_BINARY)[1]

def softLog(x):
    return softsign((logHu(x)))
    
def mnistMaxLog(x):
    maxMnist = 20.621788
    return logHu(x)/maxMnist
   
def logHu(huMoment):
    return copysign(1.0, huMoment) * log10(epsilon + abs(huMoment))

def softsign(x):
    return x/(1+abs(x))

def asGray(img):      #print(img.ndim); cv2.imshow('im', img);cv2.waitKey()
    if img.ndim != 2:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
        
def example_HuMoments():
    im1=image = cv2.imread("diamond.png")
    im2=image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    arr = cv2.HuMoments(cv2.moments(image)).flatten()
    print(arr)
    cv2.imshow('im1', im1)
    cv2.imshow('im2', im2)
    cv2.waitKey()
#example_HuMoments()  

def data_HuMoments_color():
    images, labels = load_data()
    imgray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    cv2.imshow('im1', images[0])
    cv2.imshow('im2', imgray)
    arr1 = cv2.HuMoments(cv2.moments(images[0])).flatten()
    arr2 = cv2.HuMoments(cv2.moments(imgray)).flatten()
    print(arr1)
    print(arr2)
    cv2.waitKey()
#data_HuMoments()

def printRandomImg(images, count = 10):
    randIdx = [randint(0, len(images)-1) for i in range(count)]
    [cv2.imshow('im'+str(i), images[i]) for i in randIdx]
    cv2.waitKey()

def check_load_data():
    images, _ = load_data()
    printRandomImg(images)
#check_load_data()

def check_data_HuMoments():
    images, labels = load_data()
    for i in range(0,100,10):
        cv2.imshow('im'+str(i), images[i])
        huM = getScaledHumoments(images[i])
        print(huM)
    cv2.waitKey()
#check_data_HuMoments()

def getMnistData(size=None):
    if size != None:
        return getSizedMnistData(size)
    mnist = tf.keras.datasets.mnist
    data = (x_train, y_train),(x_test, y_test) = mnist.load_data()
    return data

def getSizedMnistData(size):
    data = (x_train, y_train),(x_test, y_test) = getMnistData()
    data = (x_train, y_train),(x_test, y_test) = (x_train[:size], y_train[:size]),(x_test[:size], y_test[:size])
    return data

def __concatDataFeatures(self, xData1, xData2):
    zipped = list(zip(xData1,xData2))
    flatten = lambda x: [val for sublist in x for val in sublist]
    xData = list(map(flatten, zipped))
    return xData