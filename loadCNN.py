# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:13:23 2018

@author: tranl
"""

#from keras.models import load_model
import numpy as np
import cv2
#from SupFunctions import MiscFunc
from SupFunctions import CNNFunc
#from SupFunctions.CNNFunc import preprocessors

def main():
    image = cv2.imread(r'F:\FH_Frankfurt\Python Adrian\Working Codes\Test_CNN\Data\left\001.jpg')
    preprocessors = CNNFunc.preprocessors
    output = preprocessors.removeTop(image, 11/20)
    output1 = preprocessors.gray(output)
    output2 = preprocessors.extractWhite(output1)
    output3 = preprocessors.carBirdeyeView(output2, 5/12, 9/5, 160)
     
    
    cv2.imshow('processed', output)
    cv2.waitKey(0)
    cv2.imshow('processed', output1)
    cv2.waitKey(0)
    cv2.imshow('processed', output2)
    cv2.waitKey(0)
    cv2.imshow('processed', output3)
    cv2.waitKey(0)

if __name__ == '__main__': main()
