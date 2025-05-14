# -*- coding: UTF-8 -*-
import cv2
import numpy as np


class Line:
    """Класс для хранения информации о линии"""

    def __init__(self, color, img):
        self.color = color  # Цвет линии
        self.image = img  # Изображение линии


class LineSplitter:
    """Класс для разделения линий по цвету"""

    def __init__(self, src_img: np.ndarray):
        self.__lines = []  # Список найденных линий
        self.source_img = src_img  # Исходное изображение

    def splitter(self, color=(..., ..., ...)):
        """Выделить линии по указанному цвету с улучшенной обработкой красного"""
        hsv = cv2.cvtColor(self.source_img, cv2.COLOR_BGR2HSV)
        bgr_color = (color[2], color[1], color[0])  # Конвертация RGB в BGR

        # Получаем диапазоны цветов
        color_ranges = self.__color_range(color[0], color[1], color[2])

        # Обработка красного цвета (два диапазона)
        if len(color_ranges) == 4:
            lower1, upper1, lower2, upper2 = color_ranges
            mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower, upper = color_ranges
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        if mask.sum() == 0:
            return False

        result = cv2.bitwise_and(self.source_img, self.source_img, mask=mask)
        gray_mask = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filled_mask = np.zeros_like(gray_mask)
        cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

        interpolated_mask = self.__interpolate_mask(filled_mask)

        colored_image = np.zeros_like(self.source_img)
        colored_image[:] = bgr_color
        interpolated_result = cv2.bitwise_and(colored_image, colored_image, mask=interpolated_mask)

        self.__lines.append(Line(color, interpolated_result))
        return True

    def __interpolate_mask(self, mask, min_points_threshold=2, max_x_distance=20):
        """Интерполяция пропущенных точек в линии"""
        height, width = mask.shape
        interpolated_mask = np.copy(mask)

        for x in range(width):
            y_points = np.where(mask[:, x] == 255)[0]

            # Если точек в столбце слишком мало
            if len(y_points) < min_points_threshold:
                # Поиск ближайших заполненных столбцов слева и справа
                left_x = x - 1
                while (left_x >= 0 and
                       len(np.where(mask[:, left_x] == 255)[0]) < min_points_threshold and
                       abs(x - left_x) <= max_x_distance):
                    left_x -= 1

                right_x = x + 1
                while (right_x < width and
                       len(np.where(mask[:, right_x] == 255)[0]) < min_points_threshold and
                       abs(x - right_x) <= max_x_distance):
                    right_x += 1

                # Если найдены подходящие столбцы
                if (left_x >= 0 and right_x < width and
                        abs(x - left_x) <= max_x_distance and
                        abs(x - right_x) <= max_x_distance):
                    # Расчет средней позиции линии
                    left_y_points = np.where(mask[:, left_x] == 255)[0]
                    right_y_points = np.where(mask[:, right_x] == 255)[0]

                    left_y = int(np.mean(left_y_points))
                    right_y = int(np.mean(right_y_points))

                    # Расчет наклона линии
                    delta_x = right_x - left_x
                    delta_y = right_y - left_y
                    slope = delta_y / delta_x

                    # Интерполяция Y координаты
                    y = int(left_y + slope * (x - left_x))

                    # Расчет ширины линии
                    left_width = len(left_y_points)
                    right_width = len(right_y_points)
                    line_width = int((left_width + right_width) / 2)

                    # Заполнение интерполированной области
                    start_y = max(0, y - line_width // 2)
                    end_y = min(height, y + line_width // 2 + 1)
                    interpolated_mask[start_y:end_y, x] = 255

        return interpolated_mask


    @classmethod
    def __color_range(cls, r, g, b):
        """Определение диапазона цветов в HSV с учетом особенностей красного цвета"""
        color_srgb = np.uint8([[[r, g, b]]])
        hsv_color = cv2.cvtColor(color_srgb, cv2.COLOR_RGB2HSV)[0][0]

        # Определяем, является ли цвет красным (основная логика для красного)
        is_red = (r > 150 and g < 100 and b < 100) or \
                 (r > max(g, b) * 1.5 and r > 100)

        if is_red:
            # Для красного цвета используем два диапазона (т.к. красный на границе 0-180)
            lower1 = [0, 50, 50]  # Нижняя граница первого диапазона
            upper1 = [10, 255, 255]  # Верхняя граница первого диапазона
            lower2 = [170, 50, 50]  # Нижняя граница второго диапазона
            upper2 = [180, 255, 255]  # Верхняя граница второго диапазона
            return (lower1, upper1, lower2, upper2)
        else:
            # Для других цветов используем стандартные диапазоны
            h_ran = 15 if (r > 200 or g > 200 or b > 200) else 8  # Шире для ярких цветов
            s_ran = 40 if (r < 50 and g < 50 and b < 50) else 60  # Шире для темных цветов
            v_ran = 40

            h_range = [max(0, hsv_color[0] - h_ran), min(179, hsv_color[0] + h_ran)]
            s_range = [max(0, hsv_color[1] - s_ran), min(255, hsv_color[1] + s_ran)]
            v_range = [max(0, hsv_color[2] - v_ran), min(255, hsv_color[2] + v_ran)]

            return ([h_range[0], s_range[0], v_range[0]],
                    [h_range[1], s_range[1], v_range[1]])

    def get_lines(self):
        """Получить список найденных линий"""
        return self.__lines

    def set_src_img(self, img):
        """Установить новое исходное изображение"""
        self.source_img = img

    def cleaner(self):
        """Очистить список линий"""
        self.__lines.clear()