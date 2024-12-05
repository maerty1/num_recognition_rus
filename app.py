import argparse
from flask import Flask, render_template_string, Response, jsonify, request
import cv2
import threading
import time
import requests
from ultralytics import YOLO
import sqlite3
import os
import numpy as np
import logging
from datetime import datetime, timezone

# Настройки
DATASET_DIR = "dataset"  # Директория для сохранения данных
DB_PATH = "records.db"  # Путь к базе данных
CONFIDENCE_THRESHOLD = 0.50  # Порог уверенности для распознавания
PROCESSING_INTERVAL = 1  # Интервал обработки кадров (в секундах)
MAX_RETRY_ATTEMPTS = 5  # Максимальное количество попыток повторного подключения
NUM_SEC_FOR_SAVE_CAR_TO_DATABASE = 6  # Количество секунд для сохранения данных в базу данных
SEC_NO_DETECT_CAR = 2  # Количество секунд без детекции номера
FETCH_IMAGE_DELAY = 0.1  # Задержка при получении изображения (в секундах)
SAVE_DATASET = True  # Флаг для сохранения данных в датасет
SEARCH_TIME_WINDOW = 300  # Время поиска номера в базе (в секундах)
LOG_TO_FILE = True  # Флаг для сохранения логов в файл
LOG_FILE_PATH = "app.log"  # Путь к файлу логов
SUCCESS_RATE_THRESHOLD = 0.6  # Порог успешного распознавания (60%)
RECENT_ATTEMPTS = 5  # Количество последних попыток для оценки успешного распознавания

# Словарь для преобразования индексов классов в символы
CLASS_TO_SYMBOL = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'E', 14: 'H', 15: 'K', 16: 'M', 17: 'O', 18: 'P', 19: 'T', 20: 'X', 21: 'Y'
}

# Создание директорий для сохранения данных
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "cars"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "plate"), exist_ok=True)

app = Flask(__name__)

# Глобальные переменные для хранения состояния
detection_states = {}
source_names = []

# Глобальные переменные для хранения метрик
model_metrics = {
    'plate': {'total_frames': 0, 'detected_frames': 0, 'accuracy': 0.0},
    'symbol': {'total_frames': 0, 'detected_frames': 0, 'accuracy': 0.0}
}

# Настройка логирования
if LOG_TO_FILE:
    logging.basicConfig(level=logging.INFO, filename=LOG_FILE_PATH, filemode='a',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def create_table_if_not_exists():
    """Создает таблицы record и cameras, если они не существуют."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS record (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TEXT,
            key TEXT,
            x0 INTEGER,
            y0 INTEGER,
            x1 INTEGER,
            y1 INTEGER,
            ratio REAL,
            photo_plate TEXT,
            photo_care TEXT,
            source TEXT,
            detect_count INTEGER,
            detect_sec REAL,
            no_detect_sec REAL,
            send_to_server_code INTEGER,
            match_count INTEGER,
            time_in_view REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            x0 INTEGER NOT NULL,
            y0 INTEGER NOT NULL,
            x1 INTEGER NOT NULL,
            y1 INTEGER NOT NULL,
            name TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def migrate_table():
    """Обновляет таблицы record и cameras, добавляя новые столбцы, если они отсутствуют."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE record ADD COLUMN match_count INTEGER")
        cursor.execute("ALTER TABLE record ADD COLUMN time_in_view REAL")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Столбцы уже существуют
    finally:
        conn.close()

def fetch_image_from_url(url, delay):
    """Получает изображение по HTTP с задержкой."""
    try:
        time.sleep(delay)  # Задержка перед получением изображения
        response = requests.get(url, stream=True)
        response.raise_for_status()
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is not None:
            return frame
    except Exception as e:
        logging.error(f"Ошибка при попытке получения изображения с камеры {url}: {e}")
    return None

def process_frame(frame, plate_model, symbol_model, clahe, rect_area, source_name):
    """Обрабатывает кадр."""
    # Применение CLAHE для улучшения контраста
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Обрезаем область интереса
    x0, y0, x1, y1 = rect_area
    frame = frame[y0:y1, x0:x1]

    plate_results = plate_model(frame)
    coordinates = []
    plate_text = ""  # Инициализация переменной plate_text
    plate_img = None  # Инициализация переменной plate_img
    symbols = []  # Инициализация переменной symbols

    plate_detected = False
    symbol_detected = False

    for result in plate_results:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf[0]
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = frame[y1:y2, x1:x2]

            # Распознавание текста с использованием модели YOLO для символов
            symbol_results = symbol_model(plate_img)
            symbols = []
            for symbol_result in symbol_results:
                symbol_boxes = symbol_result.boxes
                for symbol_box in symbol_boxes:
                    symbol_confidence = symbol_box.conf[0]
                    if symbol_confidence < CONFIDENCE_THRESHOLD:
                        continue
                    symbol_x1, symbol_y1, symbol_x2, symbol_y2 = map(int, symbol_box.xyxy[0])
                    symbol_img = plate_img[symbol_y1:symbol_y2, symbol_x1:symbol_x2]
                    symbol_label = int(symbol_box.cls[0].item())  # Извлечение индекса класса
                    symbols.append((symbol_label, symbol_confidence, symbol_x1))

            # Сортировка символов по координате x
            symbols.sort(key=lambda x: x[2])
            plate_text = ''.join([CLASS_TO_SYMBOL[symbol[0]] for symbol in symbols])

            # Проверка формата распознанного текста
            if is_valid_license_plate(plate_text):
                logging.info(f"Распознанный номер с камеры {source_name}: {plate_text} (Уверенность: {confidence:.2f})")
                coordinates.append((x1, y1, x2, y2))

                plate_detected = True
                symbol_detected = True
            else:
                logging.warning(f"Неверный формат номера с камеры {source_name}: {plate_text}")

    update_metrics('plate', plate_detected)
    update_metrics('symbol', symbol_detected)

    return frame, coordinates, plate_text, plate_img, symbols

def is_valid_license_plate(plate_text):
    """Проверяет, соответствует ли распознанный текст формату российского номерного знака."""
    if len(plate_text) not in [8, 9]:
        return False
    if not plate_text[0].isalpha() or not plate_text[1:4].isdigit() or not plate_text[4:6].isalpha():
        return False
    if len(plate_text) == 9 and not plate_text[6:9].isdigit():
        return False
    if len(plate_text) == 8 and not plate_text[6:8].isdigit():
        return False
    return True

def save_image_and_data(frame, coordinates, plate_text, plate_img, symbols, save_dir, source_name):
    """Сохраняет изображения и данные для датасета."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    car_image_filename = f"{timestamp}_{plate_text}.jpg"
    car_txt_filename = f"{timestamp}_{plate_text}.txt"
    plate_image_filename = f"{timestamp}_{plate_text}_plate.jpg"
    plate_txt_filename = f"{timestamp}_{plate_text}_plate.txt"
    car_image_path = os.path.join(save_dir, "cars", car_image_filename)
    car_txt_path = os.path.join(save_dir, "cars", car_txt_filename)
    plate_image_path = os.path.join(save_dir, "plate", plate_image_filename)
    plate_txt_path = os.path.join(save_dir, "plate", plate_txt_filename)

    # Проверка наличия номера в базе данных
    if search_plate_in_db(plate_text, source_name):
        logging.info(f"Номер {plate_text} с камеры {source_name} уже существует в базе данных. Обновление match_count и time_in_view.")
        return car_image_filename, plate_image_filename

    # Сохранение исходной картинки без выделения номера
    cv2.imwrite(car_image_path, frame)

    if SAVE_DATASET:
        # Сохранение координат обнаруженной платы
        height, width, _ = frame.shape
        with open(car_txt_path, 'w') as f:
            for x1, y1, x2, y2 in coordinates:
                x_center = (x1 + x2) / (2 * width)
                y_center = (y1 + y2) / (2 * height)
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height
                f.write(f"0 {x_center} {y_center} {bbox_width} {bbox_height}\n")

    if plate_img is not None:
        # Сохранение изображения платы без выделения символов
        cv2.imwrite(plate_image_path, plate_img)

        if SAVE_DATASET:
            # Сохранение координат обнаруженных символов
            plate_height, plate_width, _ = plate_img.shape
            with open(plate_txt_path, 'w') as f:
                for symbol in symbols:
                    symbol_label, _, symbol_x1 = symbol
                    symbol_x2 = symbol_x1 + plate_width // len(symbols)
                    symbol_y1 = 0
                    symbol_y2 = plate_height
                    x_center = (symbol_x1 + symbol_x2) / (2 * plate_width)
                    y_center = (symbol_y1 + symbol_y2) / (2 * plate_height)
                    bbox_width = (symbol_x2 - symbol_x1) / plate_width
                    bbox_height = (symbol_y2 - symbol_y1) / plate_height
                    f.write(f"{symbol_label} {x_center} {y_center} {bbox_width} {bbox_height}\n")

    logging.info(f"Сохранено изображение с камеры {source_name}: {car_image_path}")
    if SAVE_DATASET:
        logging.info(f"Сохранены координаты с камеры {source_name}: {car_txt_path}")
    if plate_img is not None:
        logging.info(f"Сохранено изображение платы с камеры {source_name}: {plate_image_path}")
        if SAVE_DATASET:
            logging.info(f"Сохранены координаты символов с камеры {source_name}: {plate_txt_path}")
    return car_image_filename, plate_image_filename

def save_to_sqlite(data):
    """Сохраняет данные в SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO record (
                datetime, key, x0, y0, x1, y1, ratio,
                photo_plate, photo_care, source, detect_count,
                detect_sec, no_detect_sec, send_to_server_code,
                match_count, time_in_view
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data[0],  # datetime
            data[1],  # key
            int(data[2]),  # x0
            int(data[3]),  # y0
            int(data[4]),  # x1
            int(data[5]),  # y1
            float(data[6]),  # ratio
            data[7],  # photo_plate
            data[8],  # photo_care
            data[9],  # source
            int(data[10]),  # detect_count
            float(data[11]),  # detect_sec
            float(data[12]),  # no_detect_sec
            int(data[13]),  # send_to_server_code
            int(data[14]),  # match_count
            float(data[15])  # time_in_view
        ))
        conn.commit()
    except sqlite3.InterfaceError as e:
        logging.error(f"Ошибка записи в SQLite: {e}")
        logging.error(f"Данные: {data}")
    finally:
        conn.close()

def update_record_in_db(record_id, match_count, time_in_view):
    """Обновляет запись в базе данных."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE record SET match_count = ?, time_in_view = ? WHERE id = ?
        """, (match_count, time_in_view, record_id))
        conn.commit()
    except sqlite3.InterfaceError as e:
        logging.error(f"Ошибка обновления записи в базе данных: {e}")
    finally:
        conn.close()

def search_plate_in_db(plate_text, source_name):
    """Ищет номер в базе данных и обновляет match_count и time_in_view."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        current_time = datetime.now(timezone.utc).timestamp()
        search_time = current_time - SEARCH_TIME_WINDOW
        cursor.execute("""
            SELECT id, match_count, datetime FROM record
            WHERE key = ? AND source = ? AND datetime >= ?
        """, (plate_text, source_name, datetime.fromtimestamp(search_time, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")))

        result = cursor.fetchone()
        if result:
            record_id, match_count, last_detect_time_str = result
            last_detect_time = datetime.strptime(last_detect_time_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
            time_in_view = current_time - last_detect_time
            match_count += 1
            update_record_in_db(record_id, match_count, time_in_view)
            return True
        return False
    except sqlite3.InterfaceError as e:
        logging.error(f"Ошибка поиска в базе данных: {e}")
    finally:
        conn.close()

def fetch_cameras_from_db():
    """Извлекает данные камер из базы данных."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT url, x0, y0, x1, y1, name FROM cameras")
    cameras = cursor.fetchall()
    conn.close()
    return cameras

def capture_frame(url, plate_model, symbol_model, clahe, rect_area, source_name, detection_state, stop_event):
    """Захватывает кадр и обрабатывает его."""
    recent_plates = []
    plate_count = {}
    while not stop_event.is_set():
        frame = fetch_image_from_url(url, FETCH_IMAGE_DELAY)
        if frame is None:
            logging.warning(f"Не удалось получить изображение с камеры {source_name}. Переподключение...")
            continue

        frame, coordinates, plate_text, plate_img, symbols = process_frame(frame, plate_model, symbol_model, clahe, rect_area, source_name)

        if coordinates:  # Проверка, были ли обнаружены объекты
            recent_plates.append(plate_text)
            if len(recent_plates) > RECENT_ATTEMPTS:
                recent_plates.pop(0)

            success_rate = recent_plates.count(plate_text) / len(recent_plates)
            if success_rate >= SUCCESS_RATE_THRESHOLD:
                detection_state['detect_count'] += 1
                detection_state['no_detect_count'] = 0
                detection_state['detect_sec'] += PROCESSING_INTERVAL
                detection_state['no_detect_sec'] = 0
                detection_state['plate_text'] = plate_text  # Сохранение распознанного номера

                # Увеличиваем счетчик распознаваний для текущего номера
                if plate_text in plate_count:
                    plate_count[plate_text] += 1
                else:
                    plate_count[plate_text] = 1

                logging.info(f"Количество распознаваний номера {plate_text} с камеры {source_name}: {plate_count[plate_text]}")

                if detection_state['detect_count'] >= NUM_SEC_FOR_SAVE_CAR_TO_DATABASE:
                    car_image_filename, plate_image_filename = save_image_and_data(frame, coordinates, plate_text, plate_img, symbols, DATASET_DIR, source_name)

                    # Поиск номера в базе данных
                    if not search_plate_in_db(plate_text, source_name):
                        # Сохранение в SQLite, если номер не найден
                        datetime_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                        x0, y0, x1, y1 = coordinates[0]
                        ratio = sum([symbol[1] for symbol in symbols]) / len(symbols) if symbols else 0.0
                        save_to_sqlite((
                            datetime_str, plate_text, x0, y0, x1, y1, ratio,
                            plate_image_filename, car_image_filename, source_name, detection_state['detect_count'],
                            detection_state['detect_sec'], detection_state['no_detect_sec'], 0,
                            1, 0  # match_count, time_in_view
                        ))

                    # Сброс счетчика распознаваний для текущего номера
                    plate_count[plate_text] = 0
                    detection_state['detect_count'] = 0  # Сброс счетчика детекций
                    detection_state['detect_sec'] = 0  # Сброс времени детекций
        else:
            detection_state['no_detect_count'] += 1
            detection_state['no_detect_sec'] += PROCESSING_INTERVAL
            if detection_state['no_detect_count'] >= SEC_NO_DETECT_CAR:
                detection_state['detect_count'] = 0  # Сброс счетчика детекций
                detection_state['detect_sec'] = 0  # Сброс времени детекций

        time.sleep(PROCESSING_INTERVAL)

@app.route('/status')
def get_status():
    """Возвращает текущий статус распознавания для каждого источника в формате JSON."""
    status = []
    for url, detection_state in detection_states.items():
        source_name = source_names[list(detection_states.keys()).index(url)]
        status.append({
            'source': source_name,
            'detect_count': detection_state['detect_count'],
            'no_detect_count': detection_state['no_detect_count'],
            'detect_time': detection_state['detect_sec'],
            'no_detect_time': detection_state['no_detect_sec']
        })
    return jsonify(status)

def generate_frames(url, rect_area, source_name, detection_state):
    """Генератор для потоковой передачи кадров."""
    while True:
        frame = fetch_image_from_url(url, FETCH_IMAGE_DELAY)  # Задержка в 1 секунду
        if frame is not None:
            frame, coordinates, plate_text, _, _ = process_frame(frame, plate_model, symbol_model, clahe, rect_area, source_name)

            # Рисование рамки на изображении
            for x1, y1, x2, y2 in coordinates:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Видео с камер</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body {
            background-color: #f8f9fa;
            font-family: Arial, sans-serif;
        }
        .container {
            margin-top: 20px;
        }
        .camera-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(35%, 1fr));
            grid-gap: 20px;
            justify-content: space-around;
        }
        .camera-box {
            border: 1px solid #ddd;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            background-color: #fff;
            text-align: center;
        }
        .camera-box h6 {
            background-color: #007bff;
            color: #fff;
            padding: 10px 0;
            margin: 0;
            border-radius: 10px 10px 0 0;
        }
        .camera-box p {
            background-color: #28a745;
            color: #fff;
            padding: 5px 0;
            margin: 0;
            border-radius: 0 0 10px 10px;
        }
        .camera-box img {
            width: 100%;
            height: auto;
            border-radius: 0 0 10px 10px;
        }
        .status-container {
            margin-top: 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }
        .status-thumbnail {
            flex: 1 1 200px;
            max-width: 300px;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            background-color: #fff;
            text-align: center;
        }
        .status-thumbnail h6 {
            background-color: #007bff;
            color: #fff;
            padding: 10px;
            margin: -15px -15px 10px;
            border-radius: 10px 10px 0 0;
        }
        .status-thumbnail p {
            margin: 5px 0;
        }
    </style>
    <script>
        function updateStatus() {
            $.getJSON('/status', function(data) {
                var statusContainer = $('#status-container');
                statusContainer.empty();
                data.forEach(function(item) {
                    var statusItem = $(
                        `<div class="status-thumbnail">
                            <h6>${item.source}</h6>
                            <p>Детекций: ${item.detect_count}</p>
                            <p>Секунд без детекции: ${item.no_detect_count}</p>
                            <p>Время детекции: ${item.detect_time.toFixed(2)} с</p>
                            <p>Время без детекции: ${item.no_detect_time.toFixed(2)} с</p>
                        </div>`
                    );
                    statusContainer.append(statusItem);
                });
            });
        }

        function updatePlateText() {
            $.getJSON('/plate_text', function(data) {
                data.forEach(function(item) {
                    $(`#plate-text-${item.source_index}`).text(item.plate_text);
                });
            });
        }

        $(document).ready(function() {
            setInterval(updateStatus, 1000);
            setInterval(updatePlateText, 1000);
        });
    </script>
</head>
<body>
    <div class="container">
        <div class="camera-container">
            {% for source_name in source_names %}
                <div class="camera-box">
                    <h6 class="text-center">{{ source_name }}</h6>
                    <p class="text-center" id="plate-text-{{ loop.index0 }}"></p>
                    <img src="{{ url_for('video_feed', source_index=loop.index0) }}" width="640" height="480">
                </div>
            {% endfor %}
        </div>
        <div class="status-container" id="status-container"></div>
    </div>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    ''', source_names=source_names)
    
@app.route('/plate_text')
def get_plate_text():
    """Возвращает распознанные номера для каждой камеры."""
    plate_texts = []
    for url, detection_state in detection_states.items():
        source_name = source_names[list(detection_states.keys()).index(url)]
        plate_texts.append({
            'source_index': list(detection_states.keys()).index(url),
            'source_name': source_name,
            'plate_text': detection_state.get('plate_text', '')
        })
    return jsonify(plate_texts)

@app.route('/video_feed/<int:source_index>')
def video_feed(source_index):
    url = list(detection_states.keys())[source_index]
    rect_area = rect_cam[url]
    source_name = source_names[source_index]
    detection_state = detection_states[url]
    return Response(generate_frames(url, rect_area, source_name, detection_state), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_camera', methods=['POST'])
def add_camera():
    """Добавляет новую камеру в таблицу cameras."""
    try:
        data = request.json
        logging.info("Полученные данные: %s", data)

        url = data.get('url')
        x0 = data.get('x0')
        y0 = data.get('y0')
        x1 = data.get('x1')
        y1 = data.get('y1')
        name = data.get('name')

        if not all(v is not None and v != '' for v in [url, x0, y0, x1, y1, name]):
            return jsonify({"error": "Все поля обязательны для заполнения"}), 400

        x0, y0, x1, y1 = map(float, (x0, y0, x1, y1))

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Отладка существующих записей
        cursor.execute("SELECT name FROM cameras")
        cameras = cursor.fetchall()
        logging.info("Текущие камеры в базе данных: %s", cameras)

        # Проверка уникальности имени
        cursor.execute("SELECT * FROM cameras WHERE name = ?", (name,))
        existing_camera = cursor.fetchone()
        if existing_camera:
            conn.close()
            return jsonify({"error": "Камера с таким именем уже существует"}), 409

        # Добавление камеры
        cursor.execute("""
            INSERT INTO cameras (url, x0, y0, x1, y1, name)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (url, x0, y0, x1, y1, name))
        conn.commit()
        conn.close()

        return jsonify({"message": "Камера добавлена успешно"}), 201

    except Exception as e:
        logging.error("Ошибка: %s", str(e))
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500

@app.route('/delete_camera/<string:camera_name>', methods=['DELETE'])
def delete_camera(camera_name):
    """Удаляет камеру из таблицы cameras по её имени."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Проверяем существование камеры
        cursor.execute("SELECT * FROM cameras WHERE name = ?", (camera_name,))
        camera = cursor.fetchone()
        if not camera:
            conn.close()
            return jsonify({"error": "Камера не найдена"}), 404

        # Удаляем камеру
        cursor.execute("DELETE FROM cameras WHERE name = ?", (camera_name,))
        conn.commit()
        conn.close()

        return jsonify({"message": "Камера удалена успешно"}), 200

    except Exception as e:
        logging.error("Ошибка: %s", str(e))
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500

@app.route('/cameras', methods=['GET'])
def get_cameras():
    """Возвращает список всех камер из таблицы cameras."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cameras")
    cameras = cursor.fetchall()
    conn.close()

    cameras_list = []
    for camera in cameras:
        cameras_list.append({
            "id": camera[0],
            "url": camera[1],
            "x0": camera[2],
            "y0": camera[3],
            "x1": camera[4],
            "y1": camera[5],
            "name": camera[6]
        })

    return jsonify(cameras_list), 200

@app.route('/update_model', methods=['POST'])
def update_model():
    """Обновляет модель YOLO."""
    try:
        data = request.json
        model_type = data.get('model_type')
        model_path = data.get('model_path')

        if model_type not in ['plate', 'symbol']:
            return jsonify({"error": "Неверный тип модели. Допустимые значения: 'plate', 'symbol'"}), 400

        if not model_path:
            return jsonify({"error": "Путь к модели обязателен"}), 400

        global plate_model, symbol_model

        if model_type == 'plate':
            logging.info(f"Загрузка новой модели для плат из {model_path}...")
            plate_model = YOLO(model_path)
            logging.info("Новая модель для плат загружена.")
        elif model_type == 'symbol':
            logging.info(f"Загрузка новой модели для символов из {model_path}...")
            symbol_model = YOLO(model_path)
            logging.info("Новая модель для символов загружена.")

        return jsonify({"message": "Модель успешно обновлена"}), 200

    except Exception as e:
        logging.error("Ошибка при обновлении модели: %s", str(e))
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500

def update_metrics(model_type, detected):
    """Обновляет метрики для модели."""
    if model_type in model_metrics:
        model_metrics[model_type]['total_frames'] += 1
        if detected:
            model_metrics[model_type]['detected_frames'] += 1
        model_metrics[model_type]['accuracy'] = (model_metrics[model_type]['detected_frames'] /
                                                 model_metrics[model_type]['total_frames'])

@app.route('/model_metrics', methods=['GET'])
def get_model_metrics():
    """Возвращает метрики моделей."""
    return jsonify(model_metrics), 200

def main():
    global plate_model, symbol_model, clahe, rect_cam, source_names, detection_states

    logging.info("Загрузка модели YOLO для плат...")
    plate_model = YOLO("models/plate.pt")
    logging.info("Модель для плат загружена.")

    logging.info("Загрузка модели YOLO для символов...")
    symbol_model = YOLO("models/symbols.pt")
    logging.info("Модель для символов загружена.")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    create_table_if_not_exists()  # Создание таблицы, если она не существует
    migrate_table()  # Обновление таблицы, добавляя новые столбцы, если они отсутствуют

    stop_events = {}

    while True:
        # Извлечение данных камер из базы данных
        cameras = fetch_cameras_from_db()
        rect_cam = {}
        source_names = []
        for camera in cameras:
            url, x0, y0, x1, y1, name = camera
            rect_cam[url] = (x0, y0, x1, y1)
            source_names.append(name)

        # Словарь для хранения состояния распознавания для каждого источника
        detection_states = {url: {'detect_count': 0, 'no_detect_count': 0, 'detect_sec': 0, 'no_detect_sec': 0, 'plate_text': ''} for url in rect_cam.keys()}

        threads = []
        for url, rect_area in rect_cam.items():
            source_name_index = list(rect_cam.keys()).index(url)
            detection_state = detection_states[url]
            source_names[source_name_index] = str(source_names[source_name_index])

            # Создание или получение события остановки для камеры
            stop_event = stop_events.get(url, threading.Event())
            stop_events[url] = stop_event

            thread = threading.Thread(target=capture_frame, args=(url, plate_model, symbol_model, clahe, rect_area, source_names[source_name_index], detection_state, stop_event))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        time.sleep(PROCESSING_INTERVAL)

if __name__ == "__main__":
    threading.Thread(target=main).start()
    app.run(host='0.0.0.0', port=5000)

