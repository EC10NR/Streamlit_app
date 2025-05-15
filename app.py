import streamlit as st
import cv2
import numpy as np
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
if 'crop_state' not in st.session_state:
    st.session_state.crop_state = {
        'top_left': None,
        'bottom_right': None,
        'active': None,
        'cropped_image': None
    }
if 'original_image' not in st.session_state:
    st.session_state.original_image = None


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

        # Сбрасываем все состояния
        st.session_state.preprocessor = Preprocessor(image)
        st.session_state.processed_image = image
        st.session_state.original_image = image.copy()
        st.session_state.line_splitter = LineSplitter(image)
        st.session_state.selected_colors = []
        st.session_state.origin_point = (0, 0)
        st.session_state.min_vals = (0, 0)
        st.session_state.max_vals = (1, 1)
        st.session_state.click_origin_mode = False
        st.session_state.crop_state = {
            'top_left': None,
            'bottom_right': None,
            'active': None,
            'cropped_image': None
        }

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

    if st.sidebar.checkbox("Автоматическое выравнивание"):
        angle = st.session_state.preprocessor.auto_rotate()
        st.session_state.processed_image = st.session_state.preprocessor.get_preprocessed()
        st.session_state.line_splitter = LineSplitter(st.session_state.processed_image)
        st.sidebar.write(f"Угол поворота: {angle:.1f}°")

    if st.sidebar.checkbox("Коррекция перспективы"):
        try:
            corners = st.session_state.preprocessor.auto_correct_perspective()
            st.session_state.processed_image = st.session_state.preprocessor.get_preprocessed()
            st.session_state.line_splitter = LineSplitter(st.session_state.processed_image)
            st.sidebar.write("Перспектива скорректирована")
        except ValueError as e:
            show_warning(str(e))

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
        st.session_state.line_splitter = LineSplitter(st.session_state.processed_image)


def crop_image():
    st.header("Обрезка изображения")

    if st.session_state.processed_image is None:
        show_warning("Сначала загрузите изображение")
        return

    # Инициализация состояния обрезки
    if 'crop_points' not in st.session_state:
        st.session_state.crop_points = []
        st.session_state.crop_preview = None

    # Отображение текущего изображения (оригинал или превью)
    display_image = st.session_state.processed_image.copy()
    if st.session_state.crop_preview is not None:
        display_image = st.session_state.crop_preview

    img_display = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)

    # Обработка кликов
    coords = streamlit_image_coordinates(img_display, key="crop_coords")

    if coords is not None:
        x, y = int(coords["x"]), int(coords["y"])

        if len(st.session_state.crop_points) == 0:
            # Первый клик - верхний левый угол
            st.session_state.crop_points = [(x, y)]
            st.success(f"Верхний левый угол установлен: ({x}, {y})")

            # Создаем превью с отметкой
            preview = st.session_state.processed_image.copy()
            cv2.circle(preview, (x, y), 5, (0, 0, 255), -1)
            st.session_state.crop_preview = preview

        elif len(st.session_state.crop_points) == 1:
            # Второй клик - нижний правый угол
            first_x, first_y = st.session_state.crop_points[0]

            if x > first_x and y > first_y:
                st.session_state.crop_points.append((x, y))
                st.success(f"Нижний правый угол установлен: ({x}, {y})")

                # Создаем превью с прямоугольником
                preview = st.session_state.processed_image.copy()
                cv2.rectangle(preview,
                              st.session_state.crop_points[0],
                              st.session_state.crop_points[1],
                              (0, 255, 0), 2)
                st.session_state.crop_preview = preview
            else:
                st.error("Нижний правый угол должен быть правее и ниже верхнего левого")

    # Кнопки управления
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Подтвердить обрезку") and len(st.session_state.crop_points) == 2:
            apply_crop()

    with col2:
        if st.button("Начать заново"):
            st.session_state.crop_points = []
            st.session_state.crop_preview = None
            st.success("Выбор точек сброшен")

    with col3:
        if st.button("Отменить обрезку"):
            reset_cropping()


def apply_crop():
    """Применяет обрезку и сохраняет результат"""
    try:
        if len(st.session_state.crop_points) != 2:
            raise ValueError("Не выбраны оба угла")

        top_left = st.session_state.crop_points[0]
        bottom_right = st.session_state.crop_points[1]

        cropped = st.session_state.processed_image[
                  top_left[1]:bottom_right[1],
                  top_left[0]:bottom_right[0]]

        if cropped.size == 0:
            raise ValueError("Область обрезки пуста")

        # Сохраняем результат
        st.session_state.crop_state = {
            'top_left': top_left,
            'bottom_right': bottom_right,
            'cropped_image': cropped
        }

        st.session_state.processed_image = cropped
        st.session_state.line_splitter = LineSplitter(cropped)

        # Сбрасываем временные данные
        st.session_state.crop_points = []
        st.session_state.crop_preview = None

        st.success("Изображение успешно обрезано!")

    except Exception as e:
        st.error(f"Ошибка при обрезке: {str(e)}")


def reset_cropping():
    """Сбрасывает обрезку и восстанавливает оригинал"""
    st.session_state.processed_image = st.session_state.original_image.copy()
    st.session_state.crop_state = {
        'top_left': None,
        'bottom_right': None,
        'cropped_image': None
    }
    st.session_state.line_splitter = LineSplitter(st.session_state.processed_image)
    st.session_state.crop_points = []
    st.session_state.crop_preview = None
    st.success("Обрезка полностью сброшена")


def color_selection():
    st.header("Выбор линий по цвету")

    # Проверяем наличие изображения
    if st.session_state.crop_state.get('cropped_image') is not None:
        if 'preserved_cropped_image' not in st.session_state:
            st.session_state.preserved_cropped_image = st.session_state.crop_state['cropped_image'].copy()
        source_image = st.session_state.crop_state['cropped_image']
    elif 'preserved_cropped_image' in st.session_state:
        source_image = st.session_state.preserved_cropped_image
    elif st.session_state.processed_image is not None:
        source_image = st.session_state.processed_image
    else:
        show_warning("Сначала загрузите и обработайте изображение")
        return

    # Инициализируем или обновляем статическое изображение для отображения
    if ('color_selection_image' not in st.session_state or
        st.session_state.color_selection_image.shape != source_image.shape):
        source_rgb = cv2.cvtColor(source_image.copy(), cv2.COLOR_BGR2RGB)
        st.session_state.color_selection_image = source_rgb

    # Инициализация состояния цветов
    if 'selected_colors' not in st.session_state:
        st.session_state.selected_colors = []

    # Обработка выбора цвета через клик
    clicked_coords = streamlit_image_coordinates(st.session_state.color_selection_image, key="color_picker")

    if clicked_coords is not None:
        x, y = int(clicked_coords["x"]), int(clicked_coords["y"])
        color_rgb = st.session_state.color_selection_image[y, x]
        selected_color = (int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2]))

        if selected_color not in st.session_state.selected_colors:
            st.session_state.selected_colors.append(selected_color)
            st.success(f"Добавлен цвет: RGB{selected_color}")
        else:
            st.warning("Этот цвет уже был добавлен")

    # Ручной выбор цвета
    manual_color = st.color_picker("Или выберите цвет вручную", "#FF0000", key="manual_color_picker")

    if st.button("Добавить выбранный цвет", key="add_manual_color"):
        manual_rgb = tuple(int(manual_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
        if manual_rgb not in st.session_state.selected_colors:
            st.session_state.selected_colors.append(manual_rgb)
            st.success(f"Добавлен цвет: RGB{manual_rgb}")
        else:
            st.warning("Этот цвет уже был добавлен")

    # Отображение выбранных цветов
    if st.session_state.selected_colors:
        st.subheader("Выбранные цвета")
        cols = st.columns(5)

        for i, color in enumerate(st.session_state.selected_colors):
            with cols[i % 5]:
                color_tuple = tuple(map(int, color)) if isinstance(color, (list, np.ndarray)) else color
                text_color = "white" if sum(color_tuple) / 3 < 128 else "black"

                st.markdown(
                    f'<div style="width:100%;height:60px;background-color:rgb{color_tuple};'
                    f'margin:5px 0;border:1px solid #ddd;border-radius:5px;'
                    f'display:flex;flex-direction:column;justify-content:center;align-items:center;'
                    f'color:{text_color};">'
                    f'<div style="font-weight:bold;">Цвет {i + 1}</div>'
                    f'<div style="font-size:0.8em;">RGB{color_tuple}</div></div>',
                    unsafe_allow_html=True
                )

                if st.button(f"❌ Удалить", key=f"remove_color_{i}"):
                    st.session_state.selected_colors.pop(i)
                    st.rerun()

    # Инициализация LineSplitter с текущим изображением
    st.session_state.line_splitter = LineSplitter(source_image)

def get_current_image():
    """Возвращает текущее рабочее изображение с правильным приоритетом"""
    if st.session_state.crop_state.get('cropped_image') is not None:
        return st.session_state.crop_state['cropped_image']
    if st.session_state.processed_image is not None:
        return st.session_state.processed_image
    return st.session_state.original_image


def coordinate_system():
    st.header("Система координат")

    if 'processed_image' not in st.session_state or st.session_state.processed_image is None:
        show_warning("Сначала загрузите и обрежьте изображение")
        return

    img = st.session_state.processed_image

    if not st.session_state.get("selected_colors"):
        show_warning("Сначала выберите цвета линий в разделе 'Выбор линий по цвету'")
        return

    if "origin_point" not in st.session_state:
        st.session_state.origin_point = (0, 0)
        st.session_state.min_vals = (0, 0)

    img_display = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)

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
            show_success(f"Начало координат установлено в точке ({x}, {y})")
            st.session_state.click_origin_mode = False
            st.rerun()

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
        show_success("Точка отсчета установлена")
        st.rerun()

    st.subheader("Установка диапазона координат")
    col1, col2 = st.columns(2)
    x_min = col1.number_input("X минимальное", value=float(st.session_state.min_vals[0]), key="x_min")
    x_max = col1.number_input("X максимальное", value=float(st.session_state.max_vals[0]), key="x_max")
    y_min = col2.number_input("Y минимальное", value=float(st.session_state.min_vals[1]), key="y_min")
    y_max = col2.number_input("Y максимальное", value=float(st.session_state.max_vals[1]), key="y_max")

    if st.button("Применить диапазон координат"):
        st.session_state.min_vals = (x_min, y_min)
        st.session_state.max_vals = (x_max, y_max)
        show_success("Диапазон координат обновлен")


def line_analysis():
    st.header("Анализ линий")

    # Проверяем наличие изображения
    if st.session_state.crop_state.get('cropped_image') is not None:
        img = st.session_state.crop_state['cropped_image']
    elif 'preserved_cropped_image' in st.session_state:
        img = st.session_state.preserved_cropped_image
    elif st.session_state.processed_image is not None:
        img = st.session_state.processed_image
    else:
        show_warning("Сначала загрузите и обрежьте изображение")
        return

    if "origin_point" not in st.session_state or st.session_state.origin_point is None:
        show_warning("Сначала установите начало координат в разделе 'Система координат'")
        return

    st.session_state.lines_extracted = []
    for color in st.session_state.selected_colors:
        temp_splitter = LineSplitter(img)
        if temp_splitter.splitter(color):
            lines = temp_splitter.get_lines()
            if lines:
                st.session_state.lines_extracted.append(lines[0])

    if not st.session_state.get("lines_extracted", []):
        show_warning("Не удалось выделить линии по выбранным цветам")
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

                scanner.find_contours(filter_type=filter_type)
                scanner.calc_points(interval=interval)
                scanner.scale_points()

                points = scanner.get_scale_points()

                if not points:
                    show_warning("Не найдено ни одной точки на линии")
                    return

                st.session_state[f"points_df_{i}"] = pd.DataFrame(points, columns=["X", "Y"])
                show_success(f"Для линии {i + 1} найдено {len(points)} точек (без аппроксимации)")

            except Exception as e:
                st.error(f"Ошибка обработки линии: {str(e)}")

        if f"points_df_{i}" in st.session_state:
            df = st.session_state[f"points_df_{i}"]

            st.subheader("Результаты обработки")
            st.dataframe(df.head())

            fig = px.scatter(df, x="X", y="Y", title=f"Точки линии {i + 1}")
            st.plotly_chart(fig, use_column_width=True)

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

    upload_image()
    crop_image()  # Обрезка должна быть перед обработкой
    preprocess_controls()
    color_selection()
    coordinate_system()
    line_analysis()


if __name__ == "__main__":
    main()
