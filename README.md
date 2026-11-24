# Object Detection YOLO

Приложение для распознавания объектов на видео, фото и с веб-камеры.

## Стек

- Python
- YOLO11 (ultralytics)
- OpenCV
- PyQt6
- Pillow

## Возможности

- Выбор классов COCO для детекции
- Детекция огня по HSV-маске
- Счётчик объектов в реальном времени
- Прокрутка видео по кадрам
- Работа с RTSP-потоком и веб-камерой
- Русский интерфейс с переводом классов

## Установка

```bash
git clone https://github.com/lefords/object-detection-yolo.git
cd object-detection-yolo
pip install -r requirements.txt
