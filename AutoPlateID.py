import cv2
import os
import json
from glob import glob
from ultralytics import YOLO
import easyocr
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class LicensePlateRecognizer:
    def __init__(self):
        # Инициализация модели YOLOv8 для детекции номеров
        self.detection_model = YOLO('yolov8n.pt')  # Можно заменить на специализированную модель
        
        # Инициализация EasyOCR для распознавания текста
        self.reader = easyocr.Reader(['ru', 'en'])
        
        try:
            self.font = ImageFont.truetype("arial.ttf", 20)
        except:
            self.font = ImageFont.load_default()

    def detect_plates(self, image_path):
        """Основная функция для обработки изображения"""
        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            return None
        
        # Конвертация из BGR (OpenCV) в RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Детекция объектов с помощью YOLOv8
        results = self.detection_model(image_rgb)
        
        plates_info = []
        
        # Обработка результатов детекции
        for result in results:
            for box in result.boxes:
                # Проверяем, что это номерной знак (класс может отличаться в зависимости от модели)
                # В YOLO стандартной модели номерные знаки - это класс 2 (но лучше обучить свою модель)
                if box.cls == 2:  # Замените на нужный класс или уберите проверку, если используете специализированную модель
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Вырезаем область с номером
                    plate_roi = image[y1:y2, x1:x2]
                    
                    # Распознаем текст на номере
                    text = self.recognize_plate_text(plate_roi)
                    
                    # Добавляем информацию о номере
                    plates_info.append({
                        "box": [x1, y1, width, height],
                        "text": text
                    })
        
        return {
            "filename": os.path.basename(image_path),
            "plates": plates_info
        }
    
    def recognize_plate_text(self, plate_image):
        """Распознавание текста на номерном знаке"""
        # Улучшаем контраст для лучшего распознавания
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Используем EasyOCR для распознавания
        results = self.reader.readtext(gray, detail=0, paragraph=True)
        
        if results:
            # Объединяем все строки и удаляем лишние пробелы
            text = ' '.join(results).strip()
            # Приводим к стандартному формату (удаляем лишние символы)
            text = ''.join(c for c in text if c.isalnum() or c.isspace()).upper()
            return text
        return ""
    
    def visualize(self, image_path, output_path=None):
        """Визуализация результатов (по желанию)"""
        result = self.detect_plates(image_path)
        if not result or not result["plates"]:
            print(f"Номера не найдены на изображении: {image_path}")
            return
        
        # Открываем изображение с помощью PIL для рисования
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        for plate in result["plates"]:
            x, y, w, h = plate["box"]
            text = plate["text"]
            
            # Рисуем прямоугольник вокруг номера
            draw.rectangle([x, y, x+w, y+h], outline="red", width=2)
            
            # Добавляем текст
            draw.text((x, y-25), text, fill="red", font=self.font)
        
        if output_path:
            image.save(output_path)
            print(f"Результат сохранен в: {output_path}")
        else:
            image.show()

def process_images(input_path, output_json="results.json"):
    """Обработка одного изображения или всех изображений в папке"""
    recognizer = LicensePlateRecognizer()
    results = []
    
    # Определяем, является ли входной путь папкой или файлом
    if os.path.isdir(input_path):
        image_files = glob(os.path.join(input_path, "*.jpg")) + \
                     glob(os.path.join(input_path, "*.jpeg")) + \
                     glob(os.path.join(input_path, "*.png"))
    else:
        image_files = [input_path]
    
    # Обрабатываем каждое изображение
    for image_file in image_files:
        print(f"Обработка: {image_file}")
        result = recognizer.detect_plates(image_file)
        if result:
            results.append(result)
            
            output_img = os.path.splitext(image_file)[0] + "_result.jpg"
            recognizer.visualize(image_file, output_img)
    
    # Сохраняем результаты в JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Результаты сохранены в: {output_json}")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Распознавание автомобильных номеров")
    parser.add_argument("input", help="Путь к изображению или папке с изображениями")
    parser.add_argument("--output", help="Путь для сохранения JSON результатов", default="results.json")
    args = parser.parse_args()
    
    process_images(args.input, args.output)




# python .\AutoPlateID.py .\123.jpg