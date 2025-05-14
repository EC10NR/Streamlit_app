# -*- coding: UTF-8 -*-
import cv2
import numpy as np


class PerspectiveCorrector:
    """Класс для коррекции перспективных искажений"""

    def __init__(self, img=None):
        self.image = img  # Исходное изображение
        self.corrected_image = None  # Скорректированное изображение
        self.src_points = None  # Исходные точки (углы)
        self.dst_points = None  # Целевые точки

    def set_image(self, img):
        """Установить изображение для обработки"""
        self.image = img
        self.corrected_image = None
        self.src_points = None
        self.dst_points = None

    def auto_correct(self):
        """Автоматическая коррекция перспективы"""
        if self.image is None:
            raise ValueError("Изображение не загружено")

        # Автоматическое определение углов
        self.src_points = self._detect_corners()
        if self.src_points is None:
            raise ValueError("Не удалось автоматически определить углы графика")

        # Целевые точки (прямоугольник)
        height, width = self.image.shape[:2]
        self.dst_points = np.float32([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ])

        # Применение коррекции
        self._apply_perspective_transform()
        return self.src_points

    def manual_correct(self, src_points):
        """Ручная коррекция перспективы"""
        if self.image is None:
            raise ValueError("Изображение не загружено")
        if len(src_points) != 4:
            raise ValueError("Требуется ровно 4 точки для коррекции")

        self.src_points = np.float32(src_points)
        height, width = self.image.shape[:2]
        self.dst_points = np.float32([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ])

        self._apply_perspective_transform()

    def get_corrected_image(self):
        """Получить скорректированное изображение"""
        return self.corrected_image if self.corrected_image is not None else self.image

    def _detect_corners(self):
        """Автоматическое определение углов графика"""
        # Преобразование в градации серого
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Размытие для уменьшения шума
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Выделение краев
        edges = cv2.Canny(blurred, 50, 150)

        # Поиск контуров
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Выбор наибольшего контура (предположительно график)
            max_contour = max(contours, key=cv2.contourArea)

            # Аппроксимация контура
            epsilon = 0.02 * cv2.arcLength(max_contour, True)
            approx = cv2.approxPolyDP(max_contour, epsilon, True)

            # Если найдено 4 угла
            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
                # Сортировка углов
                corners = self._sort_corners(corners)
                return corners

        return None

    def _sort_corners(self, corners):
        """Сортировка углов: верх-левый, верх-правый, низ-правый, низ-левый"""
        sorted_by_sum = sorted(corners, key=lambda p: p[0] + p[1])
        top_left = sorted_by_sum[0]
        bottom_right = sorted_by_sum[-1]

        sorted_by_diff = sorted(corners, key=lambda p: p[0] - p[1])
        top_right = sorted_by_diff[-1]
        bottom_left = sorted_by_diff[0]

        return np.float32([top_left, top_right, bottom_right, bottom_left])

    def _apply_perspective_transform(self):
        """Применение перспективного преобразования"""
        # Вычисление матрицы преобразования
        matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

        # Применение преобразования
        height, width = self.image.shape[:2]
        self.corrected_image = cv2.warpPerspective(self.image, matrix, (width, height))