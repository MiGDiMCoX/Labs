import sys
sys.path.append('../')
import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage import data
from scipy import ndimage
import matplotlib.pyplot as plt
from utility import segmentation_utils



"""Загружаем изображение. Преобразуем в модель RGB"""

image = cv.imread('./contrast-of-the-composition.jpg')
image_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# определяем координаты начальных точек
seeds = [(360, 420), (365, 530), (370, 640),
         (310, 450), (300, 530), (310, 640), 
         (410, 1030),(410, 1000), (380, 1030),(440, 1040)]
# координаты для графика
x = list(map(lambda x: x[1], seeds))
y = list(map(lambda x: x[0], seeds))
# порог похожести цвета региона
threshold = 100
# находим сегментацию используя метод из segmentation_utils
segmented_region = segmentation_utils.region_growingHSV(image_hsv, seeds, threshold)
# накладываем маску - отображаем только участки попавшие в какой-либо сегмент
result = cv.bitwise_and(image, image, mask=segmented_region)
# Создаем белый фон того же размера, что и исходное изображение
white_background = np.ones_like(image_hsv) * 255

# Создаем итоговое изображение, где область с результатом будет видна на белом фоне
final_result = np.where(result == 0, white_background, result)
# отображаем полученное изображение
plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.scatter(x, y, marker="x", color="red", s=200)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(final_result, cv.COLOR_BGR2RGB))
plt.show()

# Алгоритм водораздела
# Бинаризируем изображение
binary_image = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
# Определяем карту расстояний
distance_map = ndimage.distance_transform_edt(binary_image)
# Определяем локальные максимумы
local_max = peak_local_max(distance_map, min_distance=20, labels=binary_image)
peaks_mask = np.zeros_like(distance_map, dtype=bool)
ind = np.array([i for i in local_max])
peaks_mask[ind[:,0], ind[:,1]] = True

"""Отобразим разные каналы полученного изображения на трехмерном графике"""

h, s, v = cv.split(image_hsv)

low_h = 0
high_h = 180

low_s = 0
high_s = 255

low_v = 0
high_v = 50

# Создание маски на основе всех трех каналов
mask = cv.inRange(h, low_h, high_h) & cv.inRange(s, low_s, high_s) & cv.inRange(v, low_v, high_v)

# 4 Каждому минимуму присваивается метка и начинается заполнение бассейнов метками
markers = ndimage.label(segmented_region, structure=np.ones((3, 3)))[0]
labels = watershed(segmented_region, segmented_region, mask=mask)
# построим результаты работы алгоритма
plt.figure(figsize=(15,20))
plt.subplot(1, 3, 1)
plt.imshow(binary_image, cmap="gray")
plt.subplot(1, 3, 2)
plt.imshow(np.uint8(distance_map + 50), cmap="gray")
plt.subplot(1, 3, 3)
plt.imshow(np.uint8(labels))
plt.show()

# Найдем границы контуров и положим в маску все кроме метки 0
mask1 = np.zeros(image.shape[0:2], dtype="uint8")
total_area = 0
for label in np.unique(labels):
    if label < 2:
        continue
    # Create a mask
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255
    mask1 = mask1 + mask

    # Find contours and determine contour area
    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv.contourArea)
    area = cv.contourArea(c)
    total_area += area
    cv.drawContours(image, [c], -1, (36,255,12), 1)

result = cv.bitwise_and(image, image, mask=mask1)

# Создаем белый фон того же размера, что и исходное изображение
white_background = np.ones_like(image_hsv) * 255

# Создаем итоговое изображение, где область с результатом будет видна на белом фоне
final_result = np.where(result == 0, white_background, result)

plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.imshow(mask1, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(final_result, cv.COLOR_BGR2RGB))
plt.show()