import cv2
import numpy as np
from core.image_rotator import ImageRotator
from core.perspective_corrector import PerspectiveCorrector
import streamlit as st


class Preprocessor:
    def __init__(self, image=None):
        if image is None and 'processed_image' in st.session_state:
            self.original_img = st.session_state.processed_image.copy()
        else:
            self.original_img = image.copy() if image is not None else np.array([])

        self.preprocessed = self.original_img.copy()
        self.current_wb = 1.0
        self.current_brightness = 0
        self.current_contrast = 1.0
        self.current_blur = 0
        self.rotator = ImageRotator()
        self.corrector = PerspectiveCorrector()

    def reset_original_img(self, img: np.ndarray):
        if img is None:
            raise ValueError("Изображение не может быть пустым")
        self.original_img = img
        self.rotator.set_image(img)
        self.corrector.set_image(img)
        self.preprocessed = self.original_img.copy()
        self.current_wb = 1.0
        self.current_brightness = 0
        self.current_contrast = 1.0
        self.current_blur = 0
        self._apply_all_effects()

    def auto_correct_perspective(self):
        """Автоматическая коррекция перспективы"""
        corners = self.corrector.auto_correct()
        self.preprocessed = self.corrector.get_corrected_image()
        self.rotator.set_image(self.preprocessed)
        self._apply_all_effects()
        return corners

    def manual_correct_perspective(self, src_points):
        """Ручная коррекция перспективы"""
        self.corrector.manual_correct(src_points)
        self.preprocessed = self.corrector.get_corrected_image()
        self.rotator.set_image(self.preprocessed)
        self._apply_all_effects()

    def auto_rotate(self):
        """Автоматическое выравнивание изображения"""
        angle = self.rotator.auto_rotate()
        self.preprocessed = self.rotator.get_rotated_image()
        self._apply_all_effects()
        return angle

    def manual_rotate(self, angle):
        """Ручной поворот изображения"""
        self.rotator.manual_rotate(angle)
        self.preprocessed = self.rotator.get_rotated_image()
        self._apply_all_effects()

    def white_balance(self, wb):
        """Коррекция баланса белого"""
        if wb < 0.1:
            wb = 0.1  # Защита от деления на ноль
        self.current_wb = wb
        self._apply_all_effects()

    def adjust_brightness(self, value):
        """Регулировка яркости"""
        self.current_brightness = value
        self._apply_all_effects()

    def adjust_contrast(self, factor):
        """Регулировка контрастности"""
        self.current_contrast = factor
        self._apply_all_effects()

    def gaussian_blur(self, kernel_size):
        """Применение размытия Гаусса"""
        if kernel_size > 0 and kernel_size % 2 == 1:  # Ядро должно быть нечетным
            self.current_blur = kernel_size
            self._apply_all_effects()

    def resize(self, scale):
        """Изменение размера изображения"""
        if scale != 0:
            width = int(self.preprocessed.shape[1] * scale)
            height = int(self.preprocessed.shape[0] * scale)
            dim = (width, height)
            resized = cv2.resize(self.preprocessed, dim, interpolation=cv2.INTER_AREA)
            self.preprocessed = resized

    def get_preprocessed(self):
        """Получить обработанное изображение"""
        return self.preprocessed if self.preprocessed.size else self.original_img

    def restore(self):
        """Восстановить исходное изображение"""
        self.preprocessed = self.original_img.copy()
        self.current_wb = 1.0
        self.current_brightness = 0
        self.current_contrast = 1.0
        self.current_blur = 0

    def _apply_all_effects(self):
        """Применить все эффекты обработки"""
        self.preprocessed = self.corrector.get_corrected_image().copy()
        self.preprocessed = self.rotator.get_rotated_image().copy()

        # Применение баланса белого
        self.preprocessed = cv2.convertScaleAbs(self.preprocessed, alpha=self.current_wb)

        # Применение яркости в HSV пространстве
        hsv = cv2.cvtColor(self.preprocessed, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        lim = 255 - self.current_brightness
        v = v.astype('int16')
        v[v > lim] = 255
        v[v <= lim] += self.current_brightness
        v = np.clip(v, 0, 255)
        v = v.astype('uint8')
        final_hsv = cv2.merge((h, s, v))
        self.preprocessed = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        # Применение контрастности
        f = 131 * (self.current_contrast - 1) / 127 + 1
        alpha_c = f
        gamma_c = 127 * (1 - f)
        self.preprocessed = cv2.addWeighted(self.preprocessed, alpha_c, self.preprocessed, 0, gamma_c)

        # Применение размытия
        if self.current_blur > 0:
            self.preprocessed = cv2.GaussianBlur(self.preprocessed, (self.current_blur, self.current_blur), 0)
