# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:29:47 2018

@author: tranl
"""

import numpy as np
import cv2
import argparse
import time

class preProcessors:
    def __init__(self, imageArray):        
        self.imageArray = imageArray
        
    def processor1(self, cropHeight, halfSquareWidth): 
        self.cropHeight = cropHeight #height to remove, can be float   
        self.halfSquareWidth = halfSquareWidth #width of the output square image, must be int
        
        imgHeight, imgWidth = self.imageArray.shape[:2]
        imgHeightLower = int(imgHeight*cropHeight)
        gray = cv2.cvtColor(self.imageArray, cv2.COLOR_BGR2GRAY)
        cropped = gray[imgHeightLower:imgHeight] #remove top 11/20 of image
        
        '''Keep only the white lanes'''
        ret, white_lanes = cv2.threshold(cropped, 200,255, cv2.THRESH_BINARY)
        wlHeight, wlWidth = white_lanes.shape[:2] 
        
        '''Dilation and Erosion to remove noise on the white lanes'''
        kernel = np.ones((5,5),np.uint8)        
        white_lanes = cv2.dilate(white_lanes, kernel,iterations = 1)
        white_lanes = cv2.erode(white_lanes, kernel,iterations = 1)
        
        '''Change perpective to birdeye's view'''
        indentX = wlWidth*5//12
        expansionY = wlHeight*9//5
        pts1 = np.float32([[0, 0], [wlWidth-1, 0], [wlWidth-1, wlHeight-1], [0, wlHeight-1]])
        pts2 = np.float32([[0, 0], [wlWidth-1, 0], [wlWidth-indentX-1, 
                           wlHeight+expansionY-1], [indentX-1, wlHeight+expansionY-1]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        mapped = cv2.warpPerspective(white_lanes, M, (wlWidth, wlHeight+expansionY))
        cropmap = mapped[(wlHeight+expansionY-halfSquareWidth*2):(wlHeight+expansionY), 
                         wlWidth//2-halfSquareWidth:wlWidth//2+halfSquareWidth]
        return cropmap
        
def main():
    #ap = argparse.ArgumentParser()
    #ap.add_argument("-d", "--data", required=True, help="Path to dataset")
    #ap.add_argument("-o", "--output", required=True, help="Path to processed dataset")
    #args = vars(ap.parse_args()) 
    
    timeBegin = time.time()
    
    for i in range(1, 61):
        image = cv2.imread(r'Data\right\{:03}.jpg'.format(i))
        processor = preProcessors(image)
        cropmap = processor.processor1(11/20, 80)
        cv2.imwrite('Output\Right\{:03}.jpg'.format(i), cropmap)
        
    timeElapsed = time.time() - timeBegin
    print('Time elapsed is:', timeElapsed)
    
    

def main2():
    #ap = argparse.ArgumentParser()
    #ap.add_argument("-d", "--data", required=True, help="Path to dataset")
    #ap.add_argument("-o", "--output", required=True, help="Path to processed dataset")
    #args = vars(ap.parse_args())    
    
    timeBegin = time.time()
    
    for i in range(1, 2):        
        #image = cv2.imread(args["data"] + "/{}.jpg".format(i))
        i2 = 25
        image = cv2.imread('Data\left\{:03}.jpg'.format(i2))
        imgHeight, imgWidth = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cropHeight = imgHeight * 11 // 20 #remove the top 11/20 of image
        #cropWidth = imgWidth * 3 // 16                    
        #cropped = gray[120:320]
        cropped = gray[cropHeight:imgHeight]
        
        '''Keep only the white lanes'''
        ret, white_lanes = cv2.threshold(cropped, 200,255, cv2.THRESH_BINARY)
        #mask_white = cv2.inRange(cropped, 200, 255)
        #white_lanes = cv2.bitwise_and(cropped, mask_white)
        wlHeight, wlWidth = white_lanes.shape[:2] 
        
        '''Dilation and Erosion to remove noise on the white lanes'''
        kernel = np.ones((5,5),np.uint8)        
        white_lanes = cv2.dilate(white_lanes, kernel,iterations = 1)
        white_lanes = cv2.erode(white_lanes, kernel,iterations = 1)    
        #dilation = cv2.dilate(erosion, kernel,iterations = 1)
        #erosion = cv2.erode(dilation, kernel,iterations = 1)  
        
        '''Change perpective to birdeye's view'''
        indentX = wlWidth*5//12
        expansionY = wlHeight*9//5
        pts1 = np.float32([[0, 0], [wlWidth-1, 0], [wlWidth-1, wlHeight-1], [0, wlHeight-1]])
        pts2 = np.float32([[0, 0], [wlWidth-1, 0], [wlWidth-indentX-1, wlHeight+expansionY-1], [indentX-1, wlHeight+expansionY-1]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        mapped = cv2.warpPerspective(white_lanes, M, (wlWidth, wlHeight+expansionY))
        #cropmap = mapped[cropHeight:(wlHeight+expansionY), cropWidth:250]
        cropmap = mapped[(wlHeight+expansionY-160):(wlHeight+expansionY), wlWidth//2-80:wlWidth//2+80]
        
        '''Print elapsed time (should spend less than 0.1s for each frame)'''
        elapsedTime = time.time()-timeBegin
        print('Elapsed time:', elapsedTime)
        
        '''Show result images'''
        #cv2.imshow('cropped', cropped)
        #cv2.imshow('white_lanes', white_lanes)
        #cv2.waitKey(0)
        #cv2.imshow('mapped', mapped)
        #cv2.waitKey(0)
        cv2.imshow('cropmap', cropmap)
        cv2.waitKey(0)
        
        

if __name__ == '__main__': main()    