# -*- coding: UTF-8 -*-
import pandas as pd
from io import BytesIO
import csv
from openpyxl import Workbook
from openpyxl.chart import ScatterChart, Reference, Series
from openpyxl.chart.axis import ChartLines


class StreamlitExporter:
    """
    Класс для экспорта данных в Streamlit-приложении с поддержкой графиков в Excel
    """

    @staticmethod
    def __format_coordinate(value):
        """
        Форматирует координату согласно заданным правилам:
        - Если значение > 1, оставляет 2 знака после запятой
        - Если значение по модулю < 1, оставляет 3 значащих цифры после запятой

        Аргументы:
            value (float): Значение координаты

        Возвращает:
            float: Отформатированное значение
        """
        abs_value = abs(value)
        if abs_value >= 1:
            return round(value, 2)
        else:
            # Для чисел меньше 1 находим первую ненулевую цифру после запятой
            s = "{0:.10f}".format(value)
            if '.' in s:
                parts = s.split('.')
                decimal_part = parts[1]
                # Находим позиции первой значащей цифры
                first_significant = None
                for i, ch in enumerate(decimal_part):
                    if ch != '0':
                        if first_significant is None:
                            first_significant = i

                if first_significant is not None:
                    cut_pos = first_significant + 3
                    formatted = parts[0] + '.' + decimal_part[:cut_pos]
                    return float(formatted)
            return value

    @staticmethod
    def __format_data(data):
        """
        Форматирует координаты в данных согласно правилам форматирования

        Аргументы:
            data: Список точек (x, y) или DataFrame

        Возвращает:
            pd.DataFrame: DataFrame с отформатированными координатами
        """
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            df = pd.DataFrame(data, columns=["X", "Y"])

        # Применяем форматирование к координатам
        df['X'] = df['X'].apply(StreamlitExporter.__format_coordinate)
        df['Y'] = df['Y'].apply(StreamlitExporter.__format_coordinate)

        return df

    @staticmethod
    def export_to_csv(data):
        """
        Экспорт данных в CSV формате без колонки Point с форматированием координат

        Аргументы:
            data: Список точек (x, y) или DataFrame

        Возвращает:
            BytesIO: Байтовый поток с CSV данными
        """
        # Форматируем координаты
        df = StreamlitExporter.__format_data(data)

        # Сортируем по X и убираем индекс
        df = df.sort_values('X').reset_index(drop=True)

        output = BytesIO()
        df.to_csv(output, index=False, encoding='utf-8')
        output.seek(0)
        return output

    @staticmethod
    def export_to_excel(data):
        """
        Экспорт данных в Excel с графиком без колонки Point с форматированием координат

        Аргументы:
            data: Список точек (x, y) или DataFrame

        Возвращает:
            BytesIO: Байтовый поток с Excel файлом
        """
        # Форматируем координаты
        df = StreamlitExporter.__format_data(data)

        # Создаем Excel файл в памяти
        output = BytesIO()
        wb = Workbook()
        ws = wb.active
        ws.title = "Data"

        # Добавляем только нужные заголовки
        ws.append(['X', 'Y'])

        # Сортируем и добавляем данные
        df = df.sort_values('X')
        for _, row in df.iterrows():
            ws.append([row['X'], row['Y']])

        # Создаем график
        chart = ScatterChart()
        chart.title = "График зависимости Y от X"
        chart.x_axis.title = "Ось X"
        chart.y_axis.title = "Ось Y"
        chart.legend = None

        # Настройки сетки
        chart.x_axis.majorGridlines = ChartLines()
        chart.y_axis.majorGridlines = ChartLines()

        # Данные для графика
        max_row = len(df) + 1
        x_values = Reference(ws, min_col=1, min_row=2, max_row=max_row)  # Столбец X
        y_values = Reference(ws, min_col=2, min_row=2, max_row=max_row)  # Столбец Y

        series = Series(y_values, x_values, title="Зависимость Y(X)")
        series.marker.symbol = "circle"
        series.marker.size = 6
        series.marker.graphicalProperties.solidFill = "4472C4"
        series.marker.graphicalProperties.line.solidFill = "4472C4"
        series.graphicalProperties.line.solidFill = "4472C4"
        series.graphicalProperties.line.width = 20000

        chart.series.append(series)
        ws.add_chart(chart, "D2")  # Смещаем график правее

        wb.save(output)
        output.seek(0)
        return output
