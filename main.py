import sys
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QLineEdit, QComboBox, QSlider,
    QCheckBox, QGroupBox, QGridLayout, QScrollArea
)

#pip install pyqt6 opencv-python ultralytics pillow numpy

COCO_CLASSES_RU = {
    0: "Люди",
    1: "Велосипеды",
    2: "Машины",
    3: "Мотоциклы",
    4: "Самолёты",
    5: "Автобусы",
    6: "Поезда",
    7: "Грузовики",
    8: "Лодки",
    9: "Светофоры",
    10: "Пожарные гидранты",
    11: "Стоп-знаки",
    12: "Паркоматы",
    15: "Кошки",
    16: "Собаки",
    17: "Лошади",
    18: "Овцы",
    19: "Коровы",
    20: "Слоны",
    21: "Медведи",
    22: "Зебры",
    23: "Жирафы",
    24: "Рюкзаки",
    25: "Зонты",
    26: "Сумки",
    27: "Галстуки",
    28: "Чемоданы",
    39: "Бутылки",
    40: "Бокалы",
    41: "Чашки",
    42: "Вилки",
    43: "Ножи",
    44: "Ложки",
    45: "Миски",
    56: "Стулья",
    57: "Диваны",
    58: "Растения",
    59: "Кровати",
    60: "Столы",
    61: "Туалеты",
    62: "Телевизоры",
    63: "Ноутбуки",
    64: "Мышки",
    65: "Пульты",
    66: "Клавиатуры",
    67: "Телефоны"
}


DEFAULT_CLASSES = {0, 1, 2, 3, 5, 7}


class VideoThread(QThread):
    frame_signal = pyqtSignal(object)
    progress_signal = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()

        self.source = None
        self.is_file = False

        self.running = False
        self.paused = False
        self.seek_frame = None

        self.model = YOLO("yolo11n.pt")

        self.selected_classes = set(DEFAULT_CLASSES)
        self.detect_fire_enabled = True

        self.font = self.load_font(24)
        self.small_font = self.load_font(22)

    def load_font(self, size):
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Arial.ttf"
        ]

        for path in font_paths:
            try:
                return ImageFont.truetype(path, size)
            except:
                pass

        return ImageFont.load_default()

    def draw_russian_text(self, frame, text, position, font=None, color=(255, 255, 255)):
        if font is None:
            font = self.small_font

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(pil_image)

        draw.text(
            position,
            text,
            font=font,
            fill=(color[2], color[1], color[0])
        )

        result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return result

    def set_source(self, source, is_file=False):
        self.source = source
        self.is_file = is_file

    def set_selected_classes(self, selected_classes):
        self.selected_classes = selected_classes

    def set_fire_enabled(self, enabled):
        self.detect_fire_enabled = enabled

    def detect_fire(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_fire = (0, 80, 120)
        upper_fire = (35, 255, 255)

        mask = cv2.inRange(hsv, lower_fire, upper_fire)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        fire_count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < 800:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = h / max(w, 1)

            if aspect < 0.5:
                continue

            fire_count += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 80, 255), 3)
            cv2.putText(
                frame,
                "FIRE",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 80, 255),
                2
            )

        return frame, fire_count

    def draw_counter(self, frame, counts, total_count, fire_count):
        x = 20
        y = 25

        panel_height = 70 + 30 * len(counts)

        if fire_count > 0:
            panel_height += 30

        cv2.rectangle(
            frame,
            (10, 10),
            (460, panel_height),
            (0, 0, 0),
            -1
        )

        frame = self.draw_russian_text(
            frame,
            f"Всего объектов: {total_count + fire_count}",
            (x, y),
            self.font,
            (255, 255, 255)
        )

        y += 35

        for class_id, count in sorted(counts.items()):
            name = COCO_CLASSES_RU.get(class_id, str(class_id))

            frame = self.draw_russian_text(
                frame,
                f"{name}: {count}",
                (x, y),
                self.small_font,
                (255, 255, 255)
            )

            y += 30

        if fire_count > 0:
            frame = self.draw_russian_text(
                frame,
                f"Огонь: {fire_count}",
                (x, y),
                self.small_font,
                (0, 120, 255)
            )

        return frame

    def run(self):
        if self.source is None:
            return

        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            print("Ошибка: источник видео не открыт")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.running = True

        while self.running:
            if self.paused:
                self.msleep(30)
                continue

            if self.seek_frame is not None and self.is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_frame)
                self.seek_frame = None

            ret, frame = cap.read()

            if not ret:
                break

            counts = {}
            total_count = 0

            if len(self.selected_classes) > 0:
                results = self.model(
                    frame,
                    classes=list(self.selected_classes),
                    conf=0.35,
                    verbose=False
                )

                result = results[0]
                annotated_frame = result.plot()

                for box in result.boxes:
                    class_id = int(box.cls[0])
                    counts[class_id] = counts.get(class_id, 0) + 1
                    total_count += 1
            else:
                annotated_frame = frame.copy()

            fire_count = 0

            if self.detect_fire_enabled:
                annotated_frame, fire_count = self.detect_fire(annotated_frame)

            annotated_frame = self.draw_counter(
                annotated_frame,
                counts,
                total_count,
                fire_count
            )

            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.progress_signal.emit(current_frame, total_frames)
            self.frame_signal.emit(annotated_frame)

            self.msleep(1)

        cap.release()

    def stop(self):
        self.running = False
        self.paused = False
        self.quit()
        self.wait()

    def pause(self):
        self.paused = True

    def play(self):
        self.paused = False

    def seek(self, frame_number):
        self.seek_frame = frame_number


class YoloApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Проект")
        self.setGeometry(200, 100, 1250, 850)

        self.file_path = None
        self.checkboxes = {}

        self.video_thread = VideoThread()
        self.video_thread.frame_signal.connect(self.update_frame)
        self.video_thread.progress_signal.connect(self.update_slider)

        self.video_label = QLabel("Видео не запущено")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumHeight(520)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #111;
                color: white;
                font-size: 22px;
                border-radius: 10px;
            }
        """)

        self.source_type = QComboBox()
        self.source_type.addItems([
            "Локальное видео",
            "Веб-камера",
            "Ссылка на камеру / видео"
        ])

        self.camera_box = QComboBox()
        self.camera_box.addItems(["0", "1", "2", "3"])

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("RTSP / HTTP / ссылка на видео")

        self.open_file_btn = QPushButton("Выбрать видео")
        self.start_btn = QPushButton("Старт")
        self.pause_btn = QPushButton("Пауза")
        self.play_btn = QPushButton("Продолжить")
        self.stop_btn = QPushButton("Стоп")

        self.select_all_btn = QPushButton("Выбрать все")
        self.clear_all_btn = QPushButton("Снять все")
        self.default_btn = QPushButton("По умолчанию")

        self.fire_checkbox = QCheckBox("Огонь")
        self.fire_checkbox.setChecked(True)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.sliderMoved.connect(self.seek_video)

        self.objects_group = QGroupBox("Объекты для распознавания")

        objects_layout = QGridLayout()

        row = 0
        col = 0

        for class_id, name in COCO_CLASSES_RU.items():
            checkbox = QCheckBox(name)
            checkbox.setChecked(class_id in DEFAULT_CLASSES)
            self.checkboxes[class_id] = checkbox

            objects_layout.addWidget(checkbox, row, col)

            col += 1

            if col >= 4:
                col = 0
                row += 1

        objects_widget = QWidget()
        objects_widget.setLayout(objects_layout)

        self.objects_scroll = QScrollArea()
        self.objects_scroll.setWidgetResizable(True)
        self.objects_scroll.setMaximumHeight(180)
        self.objects_scroll.setWidget(objects_widget)

        checkbox_buttons_layout = QHBoxLayout()
        checkbox_buttons_layout.addWidget(self.select_all_btn)
        checkbox_buttons_layout.addWidget(self.clear_all_btn)
        checkbox_buttons_layout.addWidget(self.default_btn)
        checkbox_buttons_layout.addWidget(self.fire_checkbox)

        objects_main_layout = QVBoxLayout()
        objects_main_layout.addLayout(checkbox_buttons_layout)
        objects_main_layout.addWidget(self.objects_scroll)

        self.objects_group.setLayout(objects_main_layout)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.source_type)
        controls_layout.addWidget(self.camera_box)
        controls_layout.addWidget(self.url_input)
        controls_layout.addWidget(self.open_file_btn)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.start_btn)
        buttons_layout.addWidget(self.pause_btn)
        buttons_layout.addWidget(self.play_btn)
        buttons_layout.addWidget(self.stop_btn)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.objects_group)
        main_layout.addWidget(self.slider)
        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

        self.open_file_btn.clicked.connect(self.select_file)
        self.start_btn.clicked.connect(self.start_detection)
        self.pause_btn.clicked.connect(self.pause_video)
        self.play_btn.clicked.connect(self.play_video)
        self.stop_btn.clicked.connect(self.stop_video)

        self.select_all_btn.clicked.connect(self.select_all_classes)
        self.clear_all_btn.clicked.connect(self.clear_all_classes)
        self.default_btn.clicked.connect(self.set_default_classes)

        self.source_type.currentIndexChanged.connect(self.update_ui)

        self.update_ui()

    def select_all_classes(self):
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(True)

    def clear_all_classes(self):
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(False)

    def set_default_classes(self):
        for class_id, checkbox in self.checkboxes.items():
            checkbox.setChecked(class_id in DEFAULT_CLASSES)

        self.fire_checkbox.setChecked(True)

    def get_selected_classes(self):
        selected = set()

        for class_id, checkbox in self.checkboxes.items():
            if checkbox.isChecked():
                selected.add(class_id)

        return selected

    def update_ui(self):
        source = self.source_type.currentText()

        self.camera_box.setVisible(source == "Веб-камера")
        self.url_input.setVisible(source == "Ссылка на камеру / видео")
        self.open_file_btn.setVisible(source == "Локальное видео")
        self.slider.setVisible(source == "Локальное видео")

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видео",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )

        if file_path:
            self.file_path = file_path
            self.video_label.setText(f"Выбрано видео:\n{file_path}")

    def start_detection(self):
        self.stop_video()

        source_type = self.source_type.currentText()

        if source_type == "Локальное видео":
            if not self.file_path:
                self.video_label.setText("Сначала выбери видео")
                return

            source = self.file_path
            is_file = True

        elif source_type == "Веб-камера":
            source = int(self.camera_box.currentText())
            is_file = False

        else:
            source = self.url_input.text().strip()

            if not source:
                self.video_label.setText("Вставь ссылку на камеру или видео")
                return

            is_file = False

        selected_classes = self.get_selected_classes()

        self.video_thread = VideoThread()
        self.video_thread.frame_signal.connect(self.update_frame)
        self.video_thread.progress_signal.connect(self.update_slider)

        self.video_thread.set_source(source, is_file=is_file)
        self.video_thread.set_selected_classes(selected_classes)
        self.video_thread.set_fire_enabled(self.fire_checkbox.isChecked())

        self.video_thread.start()

    def pause_video(self):
        if self.video_thread.isRunning():
            self.video_thread.pause()

    def play_video(self):
        if self.video_thread.isRunning():
            self.video_thread.play()

    def stop_video(self):
        if hasattr(self, "video_thread") and self.video_thread.isRunning():
            self.video_thread.stop()

    def seek_video(self, value):
        if self.source_type.currentText() == "Локальное видео":
            if self.video_thread.isRunning():
                self.video_thread.seek(value)

    def update_slider(self, current, total):
        if total > 0:
            self.slider.setMaximum(total)

            self.slider.blockSignals(True)
            self.slider.setValue(current)
            self.slider.blockSignals(False)

    def update_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        qt_image = QImage(
            rgb_frame.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )

        pixmap = QPixmap.fromImage(qt_image)
        pixmap = pixmap.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        )

        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.stop_video()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = YoloApp()
    window.show()

    sys.exit(app.exec())