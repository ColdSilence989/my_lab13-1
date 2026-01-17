import os
import matplotlib
matplotlib.use('Agg') # для экономии памяти
import matplotlib.pyplot as plt
import io
import base64
import requests
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
import tflite_runtime.interpreter as tflite # легкая версия, чтобы уложиться в рендер
from scipy.ndimage import gaussian_filter # по методичке

app = Flask(__name__)

# --- НАСТРОЙКИ И МОДЕЛЬ ---
# Секретный ключ для капчи (берется из настроек сервера)
RECAPTCHA_SECRET_KEY = os.environ.get('RECAPTCHA_SECRET_KEY', 'your-secret-key')

# Запускаем ResNet50
interpreter = tflite.Interpreter(model_path="models/resnet50_quant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Загружает названия предметов (1000 классов)
with open("models/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def get_base64_hist(img_np, title):
    """Рисуем гистограмму и превращаем её в текст (base64)"""
    plt.figure(figsize=(4, 2))
    for i, col in enumerate(['red', 'green', 'blue']):
        hist, _ = np.histogram(img_np[:, :, i], bins=256, range=(0, 255))
        plt.plot(hist, color=col)
    plt.title(title)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def prepare_for_ai(image_obj):
    """Умная подготовка: Center Crop (квадрат) + Resize 224x224"""
    w, h = image_obj.size
    min_dim = min(w, h)
    
    # Вычисляем координаты центрального квадрата
    left = (w - min_dim) / 2
    top = (h - min_dim) / 2
    right = (w + min_dim) / 2
    bottom = (h + min_dim) / 2
    
    # Обрезаем и сжимаем (чтобы не растянуть кота или панораму)
    img_square = image_obj.crop((left, top, right, bottom))
    img_final = img_square.resize((224, 224))
    
    # Превращаем в массив для нейронки
    input_data = np.expand_dims(img_final, axis=0).astype(np.uint8)
    return input_data

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 1. ПРОВЕРКА КАПЧИ
        token = request.form.get('g-recaptcha-response')
        verify = requests.post('https://www.google.com/recaptcha/api/siteverify', 
                               data={'secret': RECAPTCHA_SECRET_KEY, 'response': token}).json()
        if not verify.get('success'):
            return "Ошибка: Подтвердите, что вы не робот!", 403

        file = request.files.get('file')
        if not file: return "Файл не выбран", 400

        # Открываем оригинал
        original_img = Image.open(file.stream).convert('RGB')
        
        # --- МАТЕМАТИКА (ГАУСС) ---
        # Сначала превращаем в массив и размываем по методичке
        img_np = np.array(original_img)
        img_np = gaussian_filter(img_np, sigma=(1, 1, 0))

        # --- ВАРИАНТ 13: НАРЕЗКА НА 4 ЧАСТИ ---
        h, w, _ = img_np.shape
        m_h, m_w = h // 2, w // 2
        
        # Срезаем 4 прямоугольника через NumPy
        parts_np = [
            img_np[0:m_h, 0:m_w],   # 1. Лево-верх
            img_np[0:m_h, m_w:w],   # 2. Право-верх
            img_np[m_h:h, 0:m_w],   # 3. Лево-низ
            img_np[m_h:h, m_w:w]    # 4. Право-низ
        ]

        # Генерируем 5 гистограмм (4 части + 1 общая)
        hists = [get_base64_hist(p, f"Сектор {i+1}") for i, p in enumerate(parts_np)]
        full_hist = get_base64_hist(img_np, "Общая гистограмма")

        # --- НЕЙРОСЕТЬ (АНАЛИЗ КАЖДОГО УГЛА) ---
        corner_predictions = []
        for part in parts_np:
            # Превращаем кусок массива обратно в картинку для кропа
            part_img_obj = Image.fromarray(part)
            
            # Делаем Center Crop внутри сектора и скармливаем нейронке
            input_data = prepare_for_ai(part_img_obj)
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            res_idx = np.argmax(interpreter.get_tensor(output_details[0]['index'])[0])
            corner_predictions.append(labels[res_idx])

        # Анализ всей картинки целиком (тоже через Center Crop)
        full_input = prepare_for_ai(original_img)
        interpreter.set_tensor(input_details[0]['index'], full_input)
        interpreter.invoke()
        full_label = labels[np.argmax(interpreter.get_tensor(output_details[0]['index'])[0])]

        return render_template('index.html', 
                               hists=hists, 
                               full_hist=full_hist, 
                               corner_predictions=corner_predictions,
                               prediction=full_label)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)