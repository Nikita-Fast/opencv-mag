import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline


# Подготовка изображения
img_loaded = cv2.imread("./Resources/soundocre_real4.jpg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img_loaded, cv2.COLOR_BGR2RGB)
img_copy = cv2.cvtColor(img_loaded, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title("Тестовое изображение")
plt.show()


query_img = cv2.imread('./Resources/soundcore_test.webp')
train_img = img


query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw, None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None)
matcher = cv2.BFMatcher()
matches = matcher.match(queryDescriptors, trainDescriptors)
final_img = cv2.drawMatches(query_img, queryKeypoints, train_img, trainKeypoints, matches[:50], None)
plt.imshow(final_img)
plt.title("1. ORB features")
plt.show()


gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kp = sift.detect(gray_img, None)
sift_img = cv2.drawKeypoints(gray_img, kp, img_copy, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(sift_img, cmap='gray')
plt.title("2. SIFT features")
plt.show()


t_lower = 70
t_upper = 75
aperture_size = 3
L2Gradient = True

edge = cv2.Canny(img, t_lower, t_upper, apertureSize=aperture_size, L2gradient=L2Gradient)

plt.imshow(img)
plt.imshow(edge)
plt.title("3. CANNY edges")
plt.show()


gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(gray_img, cmap='gray')
plt.title("4. Перевести в grayscale")
plt.show()


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
plt.imshow(hsv)
plt.title("5. Перевести изорбражение в hsv")
plt.show()


image = cv2.flip(img, 1)
plt.imshow(image)
plt.title("6. отразить по правой границе")
plt.show()

image = cv2.flip(img, 0)
plt.imshow(image)
plt.title("7. отразить по нижней границе")
plt.show()


angle = 45 * np.pi / 180
rotate = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0]]).astype(np.float32)
rotated_img = cv2.warpAffine(img, rotate, dsize=(img.shape[1], img.shape[0]))
plt.imshow(rotated_img)
plt.title("Задание 8. Повернуть изображение на 45 градусов")
plt.show()


def rotate_around_point(px, py, angle):
    angle_rad = angle * np.pi / 180
    dx = px - (np.cos(angle_rad) * px - np.sin(angle_rad) * py)
    dy = py - (np.sin(angle_rad) * px + np.cos(angle_rad) * py)
    rot = np.array([[np.cos(angle_rad), -np.sin(angle_rad), dx], [np.sin(angle_rad), np.cos(angle_rad), dy]]).astype(np.float32)

    rotate_img = cv2.warpAffine(img, rot, dsize=(img.shape[1], img.shape[0]))

    return rotate_img


rotaded = rotate_around_point(img.shape[0] / 2, img.shape[1] / 2, 30)
plt.imshow(rotaded)
plt.title("Задание 9. Повернуть изображение на 30 градусов вокруг заданной точки")
plt.show()


shift = np.array([[1, 0, 10], [0, 1, 0]]).astype(np.float32)
shift_img = cv2.warpAffine(img, shift, dsize=(img.shape[1], img.shape[0]))
plt.imshow(shift_img)
plt.title("Задание 10. Сместить изображение но 10 пикселей вправо")
plt.show()


alpha = 1
beta = 100
new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
plt.imshow(new_image)
plt.title("Задание 11. Изменить яркость изображения")
plt.show()


# должен быть всегда больше 0, если меньше 1 - контраст уменьшается, если больше - увеличивается
alpha = 0.4
beta = 0
new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
plt.imshow(new_image)
plt.title("Задание 12. Изменить контрасть изображения")
plt.show()


gamma = 0.3
invGamma = 1.0 / gamma
table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
# apply gamma correction using the lookup table
new_image = cv2.LUT(img, table)
plt.imshow(new_image)
plt.title("Задание 13. гамма-преобразование")
plt.show()


R, G, B = cv2.split(img)

output1_R = cv2.equalizeHist(R)
output1_G = cv2.equalizeHist(G)
output1_B = cv2.equalizeHist(B)

new_image = cv2.merge((output1_R, output1_G, output1_B))

res = np.hstack((img, new_image))
plt.imshow(res)
plt.title("Задание 14. гистограммная эквализация")
plt.show()


class WarmingFilter:

    def __init__(self):
        # create look-up tables for increasing and decreasing a channel
        self.incr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 70, 140, 210, 256])
        self.decr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 30, 80, 120, 192])

    def render(self, img_rgb):
        # warming filter: increase red, decrease blue
        c_r, c_g, c_b = cv2.split(img_rgb)
        c_r = cv2.LUT(c_r, self.incr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, self.decr_ch_lut).astype(np.uint8)
        img_rgb = cv2.merge((c_r, c_g, c_b))

        # increase color saturation
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
        c_s = cv2.LUT(c_s, self.incr_ch_lut).astype(np.uint8)

        return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)

    def _create_LUT_8UC1(self, x, y):
        spl = UnivariateSpline(x, y)
        return spl(range(256))


warm_filter = WarmingFilter()
res = warm_filter.render(img)
plt.imshow(res)
plt.title("Задание 15. изменить баланс белого, сделать более теплую картинку")
plt.show()


class CoolingFilter:

    def __init__(self):
        # create look-up tables for increasing and decreasing a channel
        self.incr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 70, 140, 210, 256])
        self.decr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 30, 80, 120, 192])

    def render(self, img_rgb):
        # cooling filter: increase blue, decrease red
        c_r, c_g, c_b = cv2.split(img_rgb)
        c_r = cv2.LUT(c_r, self.decr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, self.incr_ch_lut).astype(np.uint8)
        img_rgb = cv2.merge((c_r, c_g, c_b))

        # decrease color saturation
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
        c_s = cv2.LUT(c_s, self.decr_ch_lut).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)

    def _create_LUT_8UC1(self, x, y):
        spl = UnivariateSpline(x, y)
        return spl(range(256))


cold_filter = CoolingFilter()
res = cold_filter.render(img)

plt.imshow(res)
plt.title("16. изменить баланс белого, сделать более холодную картинку")
plt.show()


gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
new_image = cv2.applyColorMap(gray_img, cv2.COLORMAP_INFERNO)
plt.imshow(new_image)
plt.title("17. изменить цветовую палитру по заданному шаблону")
plt.show()


gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
_, new_image = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
plt.imshow(new_image)
plt.title("18. бинаризация")
plt.show()


im = cv2.imread("./Resources/soundocre_real4.jpg")
gray_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
_, new_image = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(new_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im, contours, -1, (0,255,0), 3)
plt.imshow(im, cmap='gray')
plt.title("19. контуры бинаризированного изображения")
plt.show()


grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
grad = np.sqrt(grad_x**2 + grad_y**2)
grad_norm = (grad * 255 / grad.max()).astype(np.uint8)
plt.imshow(grad_norm)
plt.title("20. контуры с помощью фильтра собеля")
plt.show()


size = 13
kernel = np.ones((size, size), np.float32) / size ** 2
res = cv2.filter2D(img, -1, kernel)
plt.imshow(res)
plt.title("21. сделать размытие изображения")
plt.show()


dft = np.fft.fft2(img, axes=(0, 1))
dft_shift = np.fft.fftshift(dft)

radius = 32
lowpass_mask = np.zeros_like(img)
cy = lowpass_mask.shape[0] // 2
cx = lowpass_mask.shape[1] // 2
cv2.circle(lowpass_mask, (cx,cy), radius, (255,255,255), -1)[0]

hipass_mask = 255 - lowpass_mask

dft_shift_masked = np.multiply(dft_shift, lowpass_mask) / 255
back_ishift_masked = np.fft.ifftshift(dft_shift_masked)

dft_shift_masked_hipass = np.multiply(dft_shift, hipass_mask) / 255
back_ishift_masked_hipass = np.fft.ifftshift(dft_shift_masked_hipass)

img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0,1))
img_filtered = np.abs(img_filtered).clip(0,255).astype(np.uint8)

img_filtered_hipass = np.fft.ifft2(back_ishift_masked_hipass, axes=(0,1))
img_filtered_hipass = np.abs(img_filtered_hipass).clip(0,255).astype(np.uint8)

res = np.hstack((img_filtered_hipass, img_filtered))
plt.imshow(res)
plt.title("22-23. фильтрация частот с помощью ДПФ: Hi Pass / Low Pass")
plt.show()


size = 7
kernel = np.ones((size, size), np.uint8)
img_loaded = cv2.imread("./Resources/book.jpg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img_loaded, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.title("Тестовое изображение для демонстрации эрозии и диляции")
plt.show()

img_erosion = cv2.erode(img, kernel, iterations=1)
img_dilation = cv2.dilate(img, kernel, iterations=1)

plt.imshow(img_erosion)
plt.title("24. Эрозия")
plt.show()

plt.imshow(img_dilation)
plt.title("25. Диляция")
plt.show()
