# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:13:23 2018

@author: tranl
"""

from keras.models import load_model
import numpy as np
import cv2
#from SupFunctions import MiscFunc
from SupFunctions import CNNFunc
#from SupFunctions.CNNFunc import preprocessors

def main():
    image = cv2.imread(r'F:\FH_Frankfurt\Python Adrian\Working Codes\Test_CNN\Data\right\020.jpg')
    preprocessors = CNNFunc.preprocessors
    output = preprocessors.removeTop(image, 11/20)
    output1 = preprocessors.gray(output)
    output2 = preprocessors.extractWhite(output1)
    output3 = preprocessors.carBirdeyeView(output2, 5/12, 9/5, 160)
     
    model = load_model('streetlanes0.hdf5')
    
    processedImage = preprocessors.resizing(output3)
    processedImage = np.expand_dims(processedImage, axis = 0)
    processedImage = np.expand_dims(processedImage, axis = 3)
    #processedImage = np.expand_dims(processedImage, axis = 2)
    
    #preds = model.predict(processedImage, batch_size=32)
    prediction = model.predict(processedImage).argmax()
    if prediction == 0:
        result = 'Left'
    elif prediction == 1:
        result = 'Right'
    elif prediction == 2:
        result = 'Straight'
    else:
        result = '<ERROR> Cannot determine whether result is left, right or straight'
    print('Result is:', prediction, result)
    
    cv2.putText(image, "Label: {}".format(result),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Prediction', image)
    cv2.waitKey()
    '''cv2.imshow('processed', output)
    cv2.waitKey(0)
    cv2.imshow('processed', output1)
    cv2.waitKey(0)
    cv2.imshow('processed', output2)
    cv2.waitKey(0)
    cv2.imshow('processed', output3)
    cv2.waitKey(0)'''

if __name__ == '__main__': main()
