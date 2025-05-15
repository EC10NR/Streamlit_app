# -*- coding: UTF-8 -*-
import numpy as np
import cv2
from core.FunctionApproximator import FunctionApproximator

class Point:
    """Класс для работы с точками"""
    def __init__(self, x, y):
        self.__x = x  # Координата X
        self.__y = y  # Координата Y

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, x):
        self.__x = x

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, y):
        self.__y = y

class PointsScanner:
    """Класс для поиска и масштабирования точек на изображении"""
    def __init__(self, image=None, origin_x=0, origin_y=0):
        self.countered_img = image   # Изображение для обработки
        self.img_height = image.shape[0] if image is not None else None
        self.img_width = image.shape[1] if image is not None else None
        self.points = []             # Список найденных точек
        self.scaled_points = []      # Список масштабированных точек
        self.min_val_x = None        # Минимальное значение X
        self.min_val_y = None        # Минимальное значение Y
        self.max_val_x = None        # Максимальное значение X
        self.max_val_y = None        # Максимальное значение Y
        self.origin_x = origin_x     # Начало координат X
        self.origin_y = origin_y     # Начало координат Y
        self.points = []             # Список точек

    def avg(self, arr):
        """Вычисление среднего значения"""
        return int(np.mean(arr)) if len(arr) > 0 else 0

    def set_image(self, image):
        """Установить изображение для обработки"""
        self.countered_img = image

    def set_origin(self, origin_x, origin_y):
        """Установить начало координат"""
        self.origin_x = origin_x
        self.origin_y = origin_y

    def get_points(self):
        """Получить список точек"""
        return self.points

    def get_scale_points(self):
        """Получить масштабированные точки"""
        return self.scaled_points

    def set_minimals(self, min_vals):
        """Установить минимальные значения координат"""
        self.min_val_x = min_vals[0]
        self.min_val_y = min_vals[1]

    def set_maximals(self, max_vals):
        """Установить максимальные значения координат"""
        self.max_val_x = max_vals[0]
        self.max_val_y = max_vals[1]

    def find_contours(self, filter_type='canny', blur=False):
        """Найти контуры на изображении"""
        if self.countered_img is None:
            raise ValueError("Изображение не загружено")

        image = self.countered_img.copy()

        if blur:
            image = cv2.GaussianBlur(image, (5, 5), 4)

        if filter_type == 'canny':
            self.countered_img = cv2.Canny(image, 100, 200)
        elif filter_type == 'sobel':
            grad_x = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            grad_y = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            self.countered_img = cv2.addWeighted(abs_grad_x, 1, abs_grad_y, 1, 0)
        else:
            raise ValueError('Указан неверный тип фильтра')

        return self.countered_img

    def calc_points(self, interval=1000):
        """Вычисление координат с правильной ориентацией осей"""
        if self.countered_img is None:
            raise ValueError("Изображение не загружено")

        self.points.clear()
        x_interval = np.linspace(0, self.countered_img.shape[1] - 1, interval)
        found_points = []

        for x_pos in x_interval:
            x_pos = int(round(x_pos))
            y_pixels = np.where(self.countered_img[:, x_pos] == 255)[0]  # Правильный порядок [y,x]

            if len(y_pixels) > 0:
                y_avg = self.avg(y_pixels)
                # X увеличивается слева направо (как есть)
                x_coord = x_pos - self.origin_x
                # Y увеличивается снизу вверх (поэтому инвертируем)
                y_coord = (self.countered_img.shape[0] - y_avg) - self.origin_y
                found_points.append((x_coord, y_coord))

        self.points = found_points
        return self.points

    def scale_points(self):
        """Масштабирование с правильной ориентацией осей"""
        if not self.points:
            return

        self.scaled_points = []
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]

        x_min_pix, x_max_pix = min(x_coords), max(x_coords)
        y_min_pix, y_max_pix = min(y_coords), max(y_coords)

        for x_pix, y_pix in self.points:
            # X: слева направо
            x_scaled = self.min_val_x + (x_pix - x_min_pix) * \
                       (self.max_val_x - self.min_val_x) / (x_max_pix - x_min_pix)

            # Y: снизу вверх (не инвертируем, так как уже сделали в calc_points)
            y_scaled = self.min_val_y + (y_pix - y_min_pix) * \
                       (self.max_val_y - self.min_val_y) / (y_max_pix - y_min_pix)

            self.scaled_points.append((x_scaled, y_scaled))

    @classmethod
    def avg(cls, a):
        """Вычислить среднее значение (альтернативная реализация)"""
        return sum(a) / len(a)
