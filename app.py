# -*- coding: UTF-8 -*-
import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
import plotly.express as px
from PIL import Image
from io import BytesIO
from streamlit_image_coordinates import streamlit_image_coordinates

# Импорт основных функций
from core.preprocess import Preprocessor
from core.split import LineSplitter
from core.search import PointsScanner
from core.export import StreamlitExporter

# Настройки страницы
st.set_page_config(page_title="Извлечение данных с графиков", layout="wide")

# Инициализация состояния сессии
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = Preprocessor()
if 'line_splitter' not in st.session_state:
    st.session_state.line_splitter = None
if 'point_scanners' not in st.session_state:
    st.session_state.point_scanners = {}
if 'selected_colors' not in st.session_state:
    st.session_state.selected_colors = []
if 'origin_point' not in st.session_state:
    st.session_state.origin_point = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'min_vals' not in st.session_state:
    st.session_state.min_vals = (0, 0)
if 'max_vals' not in st.session_state:
    st.session_state.max_vals = (1, 1)
if 'click_origin_mode' not in st.session_state:
    st.session_state.click_origin_mode = False


# Вспомогательные функции
def show_warning(message):
    st.warning(message)


def show_success(message):
    st.success(message)


def display_image(image, caption):
    st.image(image, caption=caption, use_column_width=True)


def upload_image():
    st.sidebar.header("Загрузка изображения")
    uploaded_file = st.sidebar.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.session_state.preprocessor.reset_original_img(image)
        st.session_state.processed_image = image
        st.session_state.line_splitter = LineSplitter(image)
        st.session_state.selected_colors = []
        # Инициализация начальных значений координат
        st.session_state.origin_point = (0, 0)
        st.session_state.min_vals = (0, 0)
        st.session_state.max_vals = (1, 1)
        st.session_state.click_origin_mode = False
        # Очистка предыдущих результатов
        if "lines_extracted" in st.session_state:
            del st.session_state.lines_extracted
        if "points_df" in st.session_state:
            del st.session_state.points_df
        return image
    return None


def preprocess_controls():
    st.sidebar.header("Предварительная обработка")

    if st.session_state.processed_image is None:
        st.sidebar.warning("Сначала загрузите изображение")
        return

    # Основные настройки
    if st.sidebar.checkbox("Автоматическое выравнивание"):
        angle = st.session_state.preprocessor.auto_rotate()
        st.session_state.processed_image = st.session_state.preprocessor.get_preprocessed()
        st.sidebar.write(f"Угол поворота: {angle:.1f}°")

    if st.sidebar.checkbox("Коррекция перспективы"):
        try:
            corners = st.session_state.preprocessor.auto_correct_perspective()
            st.session_state.processed_image = st.session_state.preprocessor.get_preprocessed()
            st.sidebar.write("Перспектива скорректирована")
        except ValueError as e:
            show_warning(str(e))

    # Дополнительные настройки
    if st.sidebar.checkbox("Дополнительные настройки"):
        wb = st.sidebar.slider("Баланс белого", 0.1, 2.0, 1.0, 0.1)
        st.session_state.preprocessor.white_balance(wb)

        brightness = st.sidebar.slider("Яркость", -255, 255, 0)
        st.session_state.preprocessor.adjust_brightness(brightness)

        contrast = st.sidebar.slider("Контрастность", 0.0, 3.0, 1.0, 0.1)
        st.session_state.preprocessor.adjust_contrast(contrast)

        blur = st.sidebar.slider("Размытие", 0, 151, 0, 2)
        if blur > 0 and blur % 2 == 1:
            st.session_state.preprocessor.gaussian_blur(blur)

        scale = st.sidebar.slider("Масштаб", 0.1, 3.0, 1.0, 0.1)
        if scale != 0:
            st.session_state.preprocessor.resize(scale)

    if st.sidebar.button("Сбросить настройки"):
        st.session_state.preprocessor.restore()
        st.session_state.processed_image = st.session_state.preprocessor.get_preprocessed()

    st.session_state.processed_image = st.session_state.preprocessor.get_preprocessed()


def color_selection():
    st.header("Выбор линий по цвету")

    if st.session_state.processed_image is None:
        st.warning("Сначала загрузите изображение")
        return

    # Конвертируем изображение в RGB для отображения
    img_display = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)

    # Используем компонент для получения координат клика
    st.subheader("Кликните на линию для выбора цвета")
    coordinates = streamlit_image_coordinates(img_display, key="pil_picker")

    # Если был клик по изображению
    if coordinates is not None:
        x = coordinates["x"]
        y = coordinates["y"]

        # Получаем цвет пикселя и конвертируем в правильный формат
        color_bgr = st.session_state.processed_image[y, x]
        selected_color = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))  # BGR to RGB

        # Инициализируем список цветов если нужно
        if "selected_colors" not in st.session_state:
            st.session_state.selected_colors = []

        # Добавляем цвет если его еще нет
        if selected_color not in st.session_state.selected_colors:
            st.session_state.selected_colors.append(selected_color)
            st.success(f"Выбран цвет: RGB{selected_color}")
            # Сбрасываем линии при добавлении нового цвета
            if "lines_extracted" in st.session_state:
                del st.session_state.lines_extracted
        else:
            st.warning("Этот цвет уже был добавлен ранее")

    # Ручной выбор цвета
    manual_color = st.color_picker("Или выберите цвет вручную", "#FF0000")
    if st.button("Добавить выбранный цвет"):
        manual_rgb = tuple(int(manual_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
        if "selected_colors" not in st.session_state:
            st.session_state.selected_colors = []

        if manual_rgb not in st.session_state.selected_colors:
            st.session_state.selected_colors.append(manual_rgb)
            st.success(f"Добавлен цвет: RGB{manual_rgb}")
            # Сбрасываем линии при добавлении нового цвета
            if "lines_extracted" in st.session_state:
                del st.session_state.lines_extracted
        else:
            st.warning("Этот цвет уже был добавлен ранее")

    # Отображение выбранных цветов
    if "selected_colors" in st.session_state and st.session_state.selected_colors:
        st.subheader("Выбранные цвета")

        # Создаем 5 колонок для отображения цветов
        cols = st.columns(5)
        col_index = 0

        for i, color in enumerate(st.session_state.selected_colors):
            with cols[col_index]:
                # Убедимся, что цвет в правильном формате
                if isinstance(color, (list, np.ndarray)):
                    color = tuple(map(int, color))

                # Создаем квадрат с цветом
                st.markdown(
                    f'<div style="width:60px;height:60px;background-color:rgb{color};'
                    'margin:5px;border:1px solid #ccc;border-radius:5px;'
                    'display:flex;justify-content:center;align-items:center;'
                    'color:{"white" if sum(color)/3 < 128 else "black"};">'
                    f'{i + 1}</div>',
                    unsafe_allow_html=True
                )
                st.write(f"RGB{color}")

                if st.button(f"Удалить #{i + 1}", key=f"remove_{i}"):
                    st.session_state.selected_colors.pop(i)
                    st.rerun()

            col_index = (col_index + 1) % 5  # Переход на следующую колонку

    # В конце color_selection() после выбора цветов:
    if "selected_colors" in st.session_state and st.session_state.selected_colors:
        st.session_state.line_images = []
        for color in st.session_state.selected_colors:
            # Создаем временный LineSplitter для каждой линии
            splitter = LineSplitter(st.session_state.processed_image)
            splitter.splitter(color)
            lines = splitter.get_lines()
            if lines:
                # Сохраняем изображение линии в RGB формате
                line_rgb = cv2.cvtColor(lines[0].image, cv2.COLOR_BGR2RGB)
                st.session_state.line_images.append(line_rgb)


def coordinate_system():
    st.header("Система координат")

    if st.session_state.processed_image is None:
        st.warning("Сначала загрузите изображение")
        return

    if not st.session_state.get("selected_colors"):
        st.warning("Сначала выберите цвета линий в разделе 'Выбор линий по цвету'")
        return

    # Инициализация origin_point если не существует
    if "origin_point" not in st.session_state:
        st.session_state.origin_point = (0, 0)
        st.session_state.min_vals = (0, 0)

    # Конвертируем изображение в RGB для отображения
    img_display = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)

    # Выбор точки отсчета
    st.subheader("Установка начала координат")

    if st.button("Установить начало координат кликом"):
        st.session_state.click_origin_mode = True
        st.info("Кликните на изображении, чтобы установить начало координат")

    if st.session_state.click_origin_mode:
        coordinates = streamlit_image_coordinates(img_display, key="origin_picker")
        if coordinates is not None:
            x = coordinates["x"]
            y = coordinates["y"]
            st.session_state.origin_point = (x, y)
            st.session_state.min_vals = (0, 0)
            st.success(f"Начало координат установлено в точке ({x}, {y})")
            st.session_state.click_origin_mode = False
            st.rerun()

    # Ручной ввод координат
    col1, col2 = st.columns(2)
    origin_x = col1.number_input("Координата X",
                                 value=st.session_state.origin_point[0],
                                 key="origin_x")
    origin_y = col2.number_input("Координата Y",
                                 value=st.session_state.origin_point[1],
                                 key="origin_y")

    if st.button("Установить начало координат (вручную)"):
        st.session_state.origin_point = (origin_x, origin_y)
        st.session_state.min_vals = (0, 0)
        st.success("Точка отсчета установлена")
        st.experimental_rerun()

    # Диапазон координат
    st.subheader("Установка диапазона координат")
    col1, col2 = st.columns(2)
    x_min = col1.number_input("X минимальное", value=float(st.session_state.min_vals[0]), key="x_min")
    x_max = col1.number_input("X максимальное", value=float(st.session_state.max_vals[0]), key="x_max")
    y_min = col2.number_input("Y минимальное", value=float(st.session_state.min_vals[1]), key="y_min")
    y_max = col2.number_input("Y максимальное", value=float(st.session_state.max_vals[1]), key="y_max")

    if st.button("Применить диапазон координат"):
        st.session_state.min_vals = (x_min, y_min)
        st.session_state.max_vals = (x_max, y_max)
        st.success("Диапазон координат обновлен")


def line_analysis():
    st.header("Анализ линий")

    # Проверка наличия всех необходимых данных
    if not st.session_state.get("selected_colors"):
        st.warning("Сначала выберите цвета линий в разделе 'Выбор линий по цвету'")
        return

    if "processed_image" not in st.session_state:
        st.warning("Сначала загрузите и обработайте изображение")
        return

    if "origin_point" not in st.session_state or st.session_state.origin_point is None:
        st.warning("Сначала установите начало координат в разделе 'Система координат'")
        return

    # Обновляем lines_extracted
    st.session_state.lines_extracted = []
    for color in st.session_state.selected_colors:
        temp_splitter = LineSplitter(st.session_state.processed_image)
        if temp_splitter.splitter(color):
            lines = temp_splitter.get_lines()
            if lines:
                st.session_state.lines_extracted.append(lines[0])

    if not st.session_state.get("lines_extracted", []):
        st.warning("Не удалось выделить линии по выбранным цветам")
        return

    for i, line in enumerate(st.session_state.lines_extracted):
        st.subheader(f"Линия {i + 1} - RGB{line.color}")
        line_rgb = cv2.cvtColor(line.image, cv2.COLOR_BGR2RGB)
        st.image(line_rgb, caption=f"Выделенная линия {i + 1}", use_column_width=True)

        col1, col2 = st.columns(2)
        with col1:
            interval = st.number_input("Количество точек", min_value=10, max_value=5000, value=100, key=f"interval_{i}")
        with col2:
            filter_type = st.selectbox("Тип фильтра", ["canny", "sobel"], key=f"filter_{i}")

        if st.button(f"Найти точки для линии {i + 1}", key=f"process_{i}"):
            try:
                scanner = PointsScanner(
                    image=line.image.copy(),
                    origin_x=st.session_state.origin_point[0],
                    origin_y=st.session_state.origin_point[1]
                )

                scanner.set_minimals(st.session_state.min_vals)
                scanner.set_maximals(st.session_state.max_vals)

                # Находим контуры и точки без аппроксимации
                scanner.find_contours(filter_type=filter_type)
                scanner.calc_points(interval=interval)
                scanner.scale_points()

                points = scanner.get_scale_points()

                if not points:
                    st.warning("Не найдено ни одной точки на линии")
                    return

                st.session_state[f"points_df_{i}"] = pd.DataFrame(points, columns=["X", "Y"])
                st.success(f"Для линии {i + 1} найдено {len(points)} точек (без аппроксимации)")

            except Exception as e:
                st.error(f"Ошибка обработки линии: {str(e)}")


        # Отображение результатов если они есть
        if f"points_df_{i}" in st.session_state:
            df = st.session_state[f"points_df_{i}"]

            st.subheader("Результаты обработки")
            st.dataframe(df.head())

            # Визуализация
            fig = px.scatter(df, x="X", y="Y", title=f"Точки линии {i + 1}")
            st.plotly_chart(fig, use_column_width=True)

            # Экспорт данных
            st.subheader("Экспорт данных")
            export_format = st.selectbox(
                "Формат файла",
                ["CSV", "Excel"],
                key=f"export_format_{i}"
            )

            if st.button(f"Экспорт данных линии {i + 1}", key=f"export_{i}"):
                try:
                    data = st.session_state[f"points_df_{i}"]

                    if export_format == "CSV":
                        file_data = StreamlitExporter.export_to_csv(data)
                        st.download_button(
                            label="Скачать CSV",
                            data=file_data,
                            file_name=f"line_{i + 1}_data.csv",
                            mime="text/csv"
                        )
                    else:
                        file_data = StreamlitExporter.export_to_excel(data)
                        st.download_button(
                            label="Скачать Excel",
                            data=file_data,
                            file_name=f"line_{i + 1}_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                except Exception as e:
                    st.error(f"Ошибка при экспорте: {str(e)}")


def main():
    st.title("Извлечение данных с графиков")

    # Инициализация состояния для хранения кликов
    if "image_click" not in st.session_state:
        st.session_state.image_click = None

    # Обработчик кликов по изображению
    st.write("""
        <script>
            const stImage = document.querySelector("img");
            stImage.style.cursor = "crosshair";
            stImage.addEventListener("click", function(e) {
                const rect = this.getBoundingClientRect();
                const x = Math.round(e.clientX - rect.left);
                const y = Math.round(e.clientY - rect.top);
                Streamlit.setComponentValue({"x": x, "y": y});
            });
        </script>
    """, unsafe_allow_html=True)

    # Получаем данные клика
    click_data = st.session_state.get("image_click", None)
    if click_data:
        st.session_state.image_click = click_data

    # Элементы управления в боковой панели
    upload_image()
    preprocess_controls()

    # Основное содержимое
    color_selection()
    coordinate_system()
    line_analysis()


if __name__ == "__main__":
    main()
