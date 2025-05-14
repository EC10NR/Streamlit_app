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
    def export_to_csv(data):
        """
        Экспорт данных в CSV формате без колонки Point

        Аргументы:
            data: Список точек (x, y) или DataFrame

        Возвращает:
            BytesIO: Байтовый поток с CSV данными
        """
        if not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data, columns=["X", "Y"])
        else:
            df = data.copy()

        # Сортируем по X и убираем индекс
        df = df.sort_values('X').reset_index(drop=True)

        output = BytesIO()
        df.to_csv(output, index=False, encoding='utf-8')
        output.seek(0)
        return output

    @staticmethod
    def export_to_excel(data):
        """
        Экспорт данных в Excel с графиком без колонки Point

        Аргументы:
            data: Список точек (x, y) или DataFrame

        Возвращает:
            BytesIO: Байтовый поток с Excel файлом
        """
        if not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data, columns=["X", "Y"])
        else:
            df = data.copy()

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

        # Создаем график (обновляем ссылки на столбцы)
        chart = ScatterChart()
        chart.title = "График зависимости Y от X"
        chart.x_axis.title = "Ось X"
        chart.y_axis.title = "Ось Y"
        chart.legend = None

        # Настройки сетки
        chart.x_axis.majorGridlines = ChartLines()
        chart.y_axis.majorGridlines = ChartLines()

        # Данные для графика (теперь X в 1 столбце, Y во 2)
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