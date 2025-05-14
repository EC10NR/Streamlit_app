import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error


class FunctionApproximator:
    """Класс для аппроксимации функций по точкам"""

    def __init__(self, x, y):
        self.x = np.array(x)  # Координаты X
        self.y = np.array(y)  # Координаты Y

    def linear_fit(self):
        """Линейная аппроксимация: y = a * x + b"""
        coeffs = np.polyfit(self.x, self.y, 1)
        func = lambda x: coeffs[0] * x + coeffs[1]
        return func, coeffs, f"y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}"

    def polynomial_fit(self, degree):
        """Полиномиальная аппроксимация: y = a_n * x^n + ... + a_0"""
        coeffs = np.polyfit(self.x, self.y, degree)
        func = np.poly1d(coeffs)
        return func, coeffs, str(np.poly1d(coeffs))

    def power_fit(self):
        """Степенная аппроксимация: y = a * x^b"""

        def power_func(x, a, b):
            return a * x ** b

        try:
            popt, _ = curve_fit(power_func, self.x, self.y, maxfev=10000)
            func = lambda x: power_func(x, *popt)
            return func, popt, f"y = {popt[0]:.2f} * x^{popt[1]:.2f}"
        except:
            return None, None, "Степенная аппроксимация невозможна"

    def calculate_error(self, func):
        """Расчет среднеквадратичной ошибки (MSE)"""
        y_pred = func(self.x)
        return mean_squared_error(self.y, y_pred)

    def fill_missing_points(self, func, current_points, target_count):
        """Заполнение недостающих точек с использованием аппроксимированной функции"""
        if len(current_points) >= target_count:
            return current_points[:target_count]

        x_current, y_current = zip(*current_points)
        x_min, x_max = min(x_current), max(x_current)
        x_new = np.linspace(x_min, x_max, target_count)
        y_new = func(x_new)
        return list(zip(x_new, y_new))

    def auto_fit(self, max_degree=5):
        """Автоматический выбор лучшей модели аппроксимации"""
        models = {}

        # Тестирование полиномиальных моделей разных степеней
        for degree in range(1, max_degree + 1):
            func, coeffs, _ = self.polynomial_fit(degree)
            models[f'poly_{degree}'] = (func, coeffs)

        # Тестирование степенной модели (если все X положительные)
        if np.all(self.x > 0):
            func, coeffs = self.power_fit()[:2]
            if func:
                models['power'] = (func, coeffs)

        # Если модели не найдены
        if not models:
            return None, (None, None), "Не удалось найти подходящую модель"

        # Выбор модели с минимальной ошибкой
        best_model = min(models.items(), key=lambda item: self.calculate_error(item[1][0]))
        best_model_name = best_model[0]
        best_func, best_coeffs = best_model[1]

        # Форматирование результата
        if best_model_name == 'power':
            result_str = f"y = {best_coeffs[0]:.2f} * x^{best_coeffs[1]:.2f}"
        else:
            result_str = str(np.poly1d(best_coeffs))

        return best_model_name, (best_func, best_coeffs), f"Лучшая модель: {best_model_name}\nФункция: {result_str}"