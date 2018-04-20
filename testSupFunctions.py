# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:55:33 2018

@author: tranl
"""

from SupFunctions import CNNFunc
import cv2

def main():
    preprocessors = CNNFunc.preprocessors
    img = cv2.imread(r'C:\Users\tranl\JupyterNotebooks\FirstBook\index.jpg')
    gray = preprocessors.gray(img)
    brighten = preprocessors.adjustDark(gray, 1.2)
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.imshow('gray', gray)
    cv2.waitKey()
    cv2.imshow('bright', brighten)
    cv2.waitKey()
    
if __name__ == '__main__': main()