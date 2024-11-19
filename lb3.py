import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from utility import util


 


image = cv.imread('./image.jpg')
rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)






(h, w, d) = rgb_image.shape
angle = -58
scale = 1.5
center = (w // 2, h // 2)
M = cv.getRotationMatrix2D(center, angle, scale)
result_image = cv.warpAffine(rgb_image, M, (w, h))


lut = lambda i: 255 - i
result_image = lut(result_image)



gs = plt.GridSpec(1, 3)
plt.figure(figsize=(10, 8))
plt.subplot(gs[0])
plt.imshow(rgb_image)
plt.subplot(gs[1])
plt.imshow(result_image)
plt.show()