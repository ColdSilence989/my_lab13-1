import os
import matplotlib
matplotlib.use('Agg') # Экономим память (без окна)
import matplotlib.pyplot as plt
import io
import base64
import requests
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
import tflite_runtime.interpreter as tflite 
from scipy.ndimage import gaussian_filter 

app = Flask(__name__)

# --- НАСТРОЙКИ ---
RECAPTCHA_SECRET_KEY = os.environ.get('RECAPTCHA_SECRET_KEY', 'your-secret-key')
MODEL_PATH = "models/resnet50_quant.tflite"
LABELS_PATH = "models/labels.txt"

# Инициализация модели
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def image_to_base64(img_pil):
    # Если картинка слишком большая, уменьшаем её для экономии RAM
    img_pil.thumbnail((400, 400)) 
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=60) # JPEG вместо PNG
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def get_base64_hist(img_np, title):
    plt.figure(figsize=(3, 2), dpi=80) # Меньше размер и DPI
    colors = ['red', 'green', 'blue']
    for i, col in enumerate(colors):
        try:
            hist, _ = np.histogram(img_np[:, :, i], bins=64, range=(0, 255)) # Меньше бинов (64 вместо 256)
            plt.plot(hist, color=col, linewidth=1)
        except: pass
    
    plt.title(title, fontsize=8)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg')
    plt.close('all') # Полная очистка
    return base64.b64encode(buf.getvalue()).decode('utf-8')
            
    plt.title(title, fontsize=10)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def prepare_for_ai(image_obj):
    """Подготовка для нейросети: Center Crop + Resize 224x224"""
    w, h = image_obj.size
    min_dim = min(w, h)
    
    # Center Crop
    left = (w - min_dim) / 2
    top = (h - min_dim) / 2
    right = (w + min_dim) / 2
    bottom = (h + min_dim) / 2
    
    img_square = image_obj.crop((left, top, right, bottom))
    img_final = img_square.resize((224, 224))
    
    # Добавляем размерность батча (1, 224, 224, 3)
    input_data = np.expand_dims(np.array(img_final), axis=0).astype(np.uint8)
    return input_data

# --- ГЛАВНЫЙ МАРШРУТ ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 1. Проверка файла
        file = request.files.get('file')
        if not file: return "Файл не выбран", 400

        # 2. Проверка капчи (если ключ настроен)
        if RECAPTCHA_SECRET_KEY != 'your-secret-key':
            token = request.form.get('g-recaptcha-response')
            data = {'secret': RECAPTCHA_SECRET_KEY, 'response': token}
            verify = requests.post('https://www.google.com/recaptcha/api/siteverify', data=data).json()
            if not verify.get('success'):
                return "Forbidden", 403 

        # 3. Открываем оригинал
        original_img = Image.open(file.stream).convert('RGB')

        original_img.thumbnail((800, 800))
        
        # 4. Фильтр Гаусса (NumPy)
        img_np = np.array(original_img)
        blurred_np = gaussian_filter(img_np, sigma=(1, 1, 0)) # Размываем только X и Y, не цвета
        blurred_img = Image.fromarray(blurred_np)

        # 5. Нарезка на 4 части
        h, w, _ = blurred_np.shape
        cy, cx = h // 2, w // 2

        parts_np = [
            blurred_np[0:cy, 0:cx],   # 1. Верх-Лево
            blurred_np[0:cy, cx:w],   # 2. Верх-Право
            blurred_np[cy:h, 0:cx],   # 3. Низ-Лево
            blurred_np[cy:h, cx:w]    # 4. Низ-Право
        ]

        # 6. Обработка частей (Гистограммы + Картинки + AI)
        hists = []
        sector_images = []
        corner_predictions = []

        for i, part_arr in enumerate(parts_np):
            # Гистограмма
            hists.append(get_base64_hist(part_arr, f"Сектор {i+1}"))
            
            # Картинка для вывода на сайт
            part_pil = Image.fromarray(part_arr)
            sector_images.append(image_to_base64(part_pil))
            
            # Нейросеть
            input_data = prepare_for_ai(part_pil)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            res_idx = np.argmax(interpreter.get_tensor(output_details[0]['index'])[0])
            corner_predictions.append(labels[res_idx])

        # 7. Обработка полного фото
        full_hist = get_base64_hist(blurred_np, "Общая гистограмма")
        original_b64 = image_to_base64(original_img) # Оригинал для сайта
        
        full_input = prepare_for_ai(blurred_img)
        interpreter.set_tensor(input_details[0]['index'], full_input)
        interpreter.invoke()
        full_pred = labels[np.argmax(interpreter.get_tensor(output_details[0]['index'])[0])]

        # 8. Отправка в HTML
        return render_template('index.html', 
                               prediction=full_pred,
                               full_hist=full_hist,
                               
                               # Передаем списки
                               hists=hists,
                               sector_images=sector_images,
                               corner_predictions=corner_predictions,
                               
                               # Оригинал
                               original_photo=original_b64)

    return render_template('index.html')

if __name__ == '__main__':
    # Render выдает порт через переменную окружения, локально берем 10000
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
