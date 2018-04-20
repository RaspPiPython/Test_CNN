# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:45:11 2018

@author: tranl
"""

import cv2
import os
import numpy as np

class preprocessors:
    @staticmethod
    def resizing(inputImage, dimensions = (32, 32)):
        #if len(dimensions) != 2:
        #    print('<WARNING> dimensions of resizing function should have a length of 2')
        height, width = dimensions[0], dimensions[1]
        outputImage = cv2.resize(inputImage, (width, height), interpolation = cv2.INTER_AREA)
        return outputImage
    
    @staticmethod
    def removeTop(inputImage, croprate):
        height = inputImage.shape[0]
        lowerHeight = int(height*croprate)
        outputImage = inputImage[lowerHeight:height]
        return outputImage
    
    @staticmethod
    def gray(inputImage, flag = cv2.COLOR_BGR2GRAY):
        outputImage = cv2.cvtColor(inputImage, flag)
        return outputImage
    
    @staticmethod
    #input is grayscale image, output is binary image
    def extractWhite(inputImage, threshold = (200, 255)): 
        #extract white parts
        ret, outputImage = cv2.threshold(inputImage, threshold[0],  
                                         threshold[1], cv2.THRESH_BINARY)
        # closing-opening to reduce noise
        kernel = np.ones((5,5),np.uint8) 
        outputImage = cv2.morphologyEx(outputImage, cv2.MORPH_CLOSE, kernel)
        #outputImage = cv2.morphologyEx(outputImage, cv2.MORPH_OPEN, kernel)
        return outputImage
    
    @staticmethod
    # change perspective of input image to birdeye view and keep a square in the middle of bottom side
    def carBirdeyeView(inputImage, bottomShrink, heightExpand, squareWidth):
        height, width = inputImage.shape[:2]
        halfSquareWidth = squareWidth//2
        wShrink = int(width*bottomShrink)
        hExpand = int(height*heightExpand)
        pts1 = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
        pts2 = np.float32([[0, 0], [width-1, 0], 
                           [width-wShrink-1, height+hExpand-1], 
                           [wShrink-1, height+hExpand-1]])
        persMatrix = cv2.getPerspectiveTransform(pts1, pts2)
        outputImage = cv2.warpPerspective(inputImage, persMatrix, (width, height+hExpand))
        outputImage = outputImage[(height+hExpand-squareWidth):(height+hExpand), 
                         width//2-halfSquareWidth:width//2+halfSquareWidth]
        return outputImage
    
    '''not tested yet'''
    @staticmethod
    # increase brightness of image if it is too dark, gamma<1 darker, gamma>1 brighter
    def adjustDark(inputImage, gamma=1.0):
        maxValue = np.max(inputImage)
        if maxValue < 250:
            invGamma = 1/((2-maxValue/255)*gamma)
            table = np.array([(i/255)**invGamma*255 for i in range(0, 256)]).astype('uint8')
            return cv2.LUT(inputImage, table)
        else:
            return inputImage
    

# This paths0 function extract all file paths from the directory        
def paths0(directory):
    fileNames = [f for f in os.listdir(directory)]
    filePaths = []
    for fileName in fileNames:
        filePath = '\\'.join((directory, fileName))
        filePaths.append(filePath)
    return filePaths

# This paths function assume that inside directory, there are folders with names 
# the same as classes and images of each class will be inside their respective folders
def paths(directory):
    classNames = [f for f in os.listdir(directory)]
    classPaths = []
    filePaths = []
    for className in classNames:
        classPath = '\\'.join((directory, className))
        classPaths.append(classPath)
    for classPath in classPaths:
        fileNames = [f for f in os.listdir(classPath)]
        for fileName in fileNames:
            filePath = '\\'.join((classPath, fileName))
            filePaths.append(filePath)
    return filePaths        

# Attach labels on the images
def loadProccessedLabels(filePaths): 
    (images, labels) = ([], [])
    for filePath in filePaths:
        image = cv2.imread(filePath, 0) #read img as gray scale img
        label = filePath.split(os.path.sep)[-2] 
        image = preprocessors.resizing(image, (32, 32))
        images.append(image)
        labels.append(label)
    return (np.array(images), np.array(labels))
