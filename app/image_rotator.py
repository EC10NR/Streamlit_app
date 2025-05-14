# -*- coding: UTF-8 -*-
import cv2
import numpy as np


class ImageRotator:
    """Класс для выравнивания изображений"""

    def __init__(self, img=None):
        self.image = img  # Исходное изображение
        self.rotated_image = None  # Повернутое изображение
        self.angle = 0.0  # Угол поворота

    def set_image(self, img):
        """Установить изображение для обработки"""
        self.image = img
        self.rotated_image = None
        self.angle = 0.0

    def auto_rotate(self):
        """Автоматическое выравнивание изображения"""
        if self.image is None:
            raise ValueError("Изображение не загружено")

        # Определение угла наклона
        self.angle = self._detect_angle()

        # Поворот изображения
        self._rotate_image(self.angle)
        return self.angle

    def manual_rotate(self, angle):
        """Ручной поворот изображения"""
        if self.image is None:
            raise ValueError("Изображение не загружено")

        self.angle = angle
        self._rotate_image(angle)

    def get_rotated_image(self):
        """Получить повернутое изображение"""
        return self.rotated_image if self.rotated_image is not None else self.image

    def _detect_angle(self):
        """Определить угол наклона изображения"""
        # Преобразование в градации серого
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Размытие для уменьшения шума
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Выделение краев
        edges = cv2.Canny(blurred, 50, 150)

        # Поиск линий с помощью преобразования Хафа
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is not None:
            angles = []
            for rho, theta in lines[0]:
                # Преобразование радиан в градусы
                angle = (theta * 180 / np.pi) - 90
                if -45 <= angle <= 45:  # Ограничение разумных значений
                    angles.append(angle)

            if angles:
                return np.mean(angles)  # Средний угол наклона

        return 0.0  # Если линии не найдены

    def _rotate_image(self, angle):
        """Повернуть изображение на заданный угол"""
        height, width = self.image.shape[:2]
        center = (width // 2, height // 2)

        # Матрица поворота
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Расчет новых размеров изображения
        cos_val = np.abs(np.cos(np.radians(angle)))
        sin_val = np.abs(np.sin(np.radians(angle)))
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))

        # Корректировка матрицы поворота
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        # Применение поворота
        self.rotated_image = cv2.warpAffine(self.image, rotation_matrix, (new_width, new_height))