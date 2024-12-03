import sys
sys.path.append('../')
# %matplotlib inline
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from utility import util
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb





"""Загружаем изображение. Преобразуем в модель RGB"""

image = cv.imread('./contrast-of-the-composition.jpg')
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.show()


r, g, b = cv.split(image_rgb)




"""Преобразуем изображение в цветовую модель HSV"""

image_hsv = cv.cvtColor(image_rgb, cv.COLOR_RGB2HSV)

"""Отобразим разные каналы полученного изображения на трехмерном графике"""

h, s, v = cv.split(image_hsv)


#lower_black = np.array([0, 0, 0])  # Нижняя граница
#upper_black = np.array([180, 255, 50])  # Верхняя граница


low_h = 0
high_h = 255

low_s = 0
high_s = 45

low_v = 0
high_v = 205

# Создание маски на основе всех трех каналов
mask = cv.inRange(h, low_h, high_h) & cv.inRange(s, low_s, high_s) & cv.inRange(v, low_v, high_v)
result = cv.bitwise_and(image_rgb, image_rgb, mask=mask)

gs = plt.GridSpec(1, 2)
plt.figure(figsize=(10, 8))


plt.subplot(gs[0])
plt.imshow(mask, cmap='gray')
plt.title('Маска')
plt.xticks([]), plt.yticks([])


plt.subplot(gs[1])
plt.imshow(result)
plt.title('Изображение с пикселями выделенного цвета')
plt.show()
