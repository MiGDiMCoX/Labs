import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from utility import util

def ahe(arr,hist):
    shape = arr.shape
    result = np.zeros(shape,dtype=int)
    range1 = range(shape[0]-1)
    range2 = range(shape[1]-1)
    
    matrix = np.zeros(hist.shape)
    
    for i in range(255):
        for j in range(i):
            matrix[i] += hist[j]
    
    for i in range1:
        for j in range2:
                result[i][j] = matrix[arr[i][j]]
                
    return result

 


image = cv.imread('./winter_cat.png')

gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)



channels = [0]
histSize = [256]
irange = [0, 256]


hist = cv.calcHist([gray_image], channels, None, histSize, irange)


lut = lambda i: 255 * (ahe(i,hist)/sum(hist))
result_image = lut(gray_image)

gs = plt.GridSpec(2, 2)
plt.figure(figsize=(10, 8))
plt.subplot(gs[0])
plt.imshow(gray_image, cmap='gray')
plt.subplot(gs[1])
plt.imshow(result_image, cmap='gray')
plt.subplot(gs[2])
plt.hist(gray_image.reshape(-1), 256, irange)
plt.subplot(gs[3])
plt.hist(result_image.reshape(-1), 256, irange)
plt.show()
