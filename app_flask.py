import sys
sys.path.insert(0, r'D:\Bishe\yolov5')   # 注意使用原始字符串或双反斜杠
import io
import base64
import time
import sqlite3
from datetime import datetime
from flask import Flask, request, render_template_string
import torch
from PIL import Image
import cv2
import numpy as np
from paddleocr import PaddleOCR
import torch

app = Flask(__name__)

# ----------  加载 YOLO 模型 ----------
from models.common import AutoShape, DetectMultiBackend

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend('D:/Bishe/yolov5/runs/train/exp42/weights/best.pt', device=device)
model = AutoShape(model)
model.conf = 0.5
# ----------  初始化 OCR ----------
ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
print("OCR 初始化完成")


# ----------  数据库初始化 ----------
def init_db():
    conn = sqlite3.connect('license_plate.db')
    cursor = conn.cursor()
    # 识别记录表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recognition_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            plate_text TEXT,
            confidence REAL,
            processing_time REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # 车辆信息表（含是否允许通行字段）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS license_plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT UNIQUE,
            owner_name TEXT,
            plate_color TEXT,
            vehicle_type TEXT,
            notes TEXT,
            is_allowed INTEGER DEFAULT 1
        )
    ''')
    # 为旧表增加 is_allowed 列（如果已存在则忽略）
    try:
        cursor.execute("ALTER TABLE license_plates ADD COLUMN is_allowed INTEGER DEFAULT 1")
    except:
        pass
    conn.commit()
    conn.close()
    print("数据库初始化完成")


def insert_sample_vehicles():
    """插入一些示例车辆数据"""
    conn = sqlite3.connect('license_plate.db')
    cursor = conn.cursor()
    samples = [
        ('京A12345', '张三', 1),
        ('沪B67890', '李四', 0),
        ('粤CD12345', '王五', 1),
        ('苏E88888', '赵六', 1),
    ]
    for plate, owner, allowed in samples:
        cursor.execute("INSERT OR IGNORE INTO license_plates (plate_number, owner_name, is_allowed) VALUES (?, ?, ?)",
                       (plate, owner, allowed))
    conn.commit()
    conn.close()
    print("示例车辆数据已插入")


def save_recognition(image_name, plate_text, confidence, processing_time):
    conn = sqlite3.connect('license_plate.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO recognition_records (image_name, plate_text, confidence, processing_time)
        VALUES (?, ?, ?, ?)
    ''', (image_name, plate_text, confidence, processing_time))
    conn.commit()
    conn.close()


init_db()
insert_sample_vehicles()


# ---------- 车牌图像增强（提高 OCR 识别率）----------
def enhance_plate_image(plate_pil):
    """对裁剪的车牌区域进行预处理，返回处理后的 numpy 灰度图"""
    # 转为灰度图
    img = np.array(plate_pil.convert('L'))
    # 直方图均衡化（增强对比度）
    img = cv2.equalizeHist(img)
    # 高斯滤波去噪
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # 自适应阈值二值化（突出字符）
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    # 放大到原尺寸的2倍（小字识别更好）
    h, w = img.shape
    img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    return img


# ---------- 检测+识别+通行判断 ----------
def detect_plate(image_bytes, image_name="unknown.jpg"):
    start_time = time.time()
    # 读取图片
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # YOLO 检测
    results = model(image)
    detections = results.pandas().xyxy[0]

    # 绘制结果图（base64）
    annotated = results.render()[0]
    pil_img = Image.fromarray(annotated)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    if not detections.empty:
        info = f"检测到 {len(detections)} 个车牌<br>"
        for idx, row in detections.iterrows():
            x1, y1, x2, y2 = int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax)
            conf = row.confidence
            # 裁剪车牌区域
            plate_roi = image.crop((x1, y1, x2, y2))

            # ========== 直接使用原始区域（不增强）==========
            try:
                ocr_result = ocr.ocr(np.array(plate_roi))
                if ocr_result and ocr_result[0]:
                    plate_text = ocr_result[0][0][1][0]
                else:
                    plate_text = "识别失败"
            except Exception as e:
                print(f"OCR 错误: {e}")
                plate_text = "OCR出错"

            # 保存识别记录
            processing_time = time.time() - start_time
            save_recognition(image_name, plate_text, float(conf), processing_time)

            # 查询数据库判断是否允许通行
            conn = sqlite3.connect('license_plate.db')
            cursor = conn.cursor()
            cursor.execute("SELECT owner_name, is_allowed FROM license_plates WHERE plate_number=?", (plate_text,))
            row_db = cursor.fetchone()
            conn.close()
            if row_db:
                owner_name, is_allowed = row_db
                status = "√允许通行" if is_allowed == 1 else "×禁止通行"
                extra = f"（{status}，车主：{owner_name}）"
            else:
                extra = "（未注册车辆，需人工核实）"

            info += f"车牌 {idx + 1}: <b>{plate_text}</b> (置信度 {conf:.2f})<br>{extra}<br>"
    else:
        info = "未检测到车牌，请换一张图片试试"


    return img_base64, info


# ---------- 6. HTML （主页 + 管理页面）----------
HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>车牌检测通行系统</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
        .container { max-width: 800px; margin: auto; border: 1px solid #ccc; padding: 20px; border-radius: 10px; }
        input { margin: 10px; }
        .result-img { max-width: 100%; margin-top: 20px; }
        .info { background: #f0f0f0; padding: 10px; border-radius: 5px; margin-top: 20px; text-align: left; }
        .nav { margin-top: 20px; }
        .nav a { margin: 0 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1> 车牌检测通行系统</h1>
        <form method="post" enctype="multipart/form-data" action="/">
            <input type="file" name="file" accept="image/jpeg,image/png,image/jpg" required>
            <input type="submit" value="上传并检测">
        </form>
        <div class="nav">
            <a href="/admin">管理车辆白名单</a>
            <a href="/records">查看识别记录</a>
        </div>
        {% if image_data %}
            <h3>检测结果</h3>
            <img class="result-img" src="data:image/jpeg;base64,{{ image_data }}" alt="检测结果">
            <div class="info">
                <strong>识别信息：</strong><br>
                {{ info_text | safe }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

ADMIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>车辆黑白名单管理</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; margin-top: 20px; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        .message { color: red; }
    </style>
</head>
<body>
    <h1>车辆黑白名单管理</h1>
    <form method="post">
        <label>车牌号: <input type="text" name="plate_number" required></label><br>
        <label>车主姓名: <input type="text" name="owner_name"></label><br>
        <label>允许通行: <input type="checkbox" name="is_allowed" checked></label><br>
        <input type="submit" value="添加/更新">
    </form>
    <p class="message">{{ message }}</p>
    <h2>现有车辆</h2>
    <table>
        <tr><th>车牌号</th><th>车主</th><th>状态</th><th>操作</th></tr>
        {% for v in vehicles %}
        <tr>
            <td>{{ v[0] }}</td>
            <td>{{ v[1] if v[1] else '' }}</td>
            <td>{{ '允许' if v[2]==1 else ' 禁止' }}</td>
            <td><a href="/delete_vehicle?plate={{ v[0] }}">删除</a></td>
        </tr>
        {% endfor %}
    </table>
    <p><a href="/">返回检测页面</a></p>
</body>
</html>
"""

RECORDS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>识别记录</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
    </style>
</head>
<body>
    <h1>识别记录</h1>
    <form method="get">
        <input type="text" name="plate" placeholder="输入车牌号筛选" value="{{ plate_filter }}">
        <button type="submit">搜索</button>
    </form>
    <table>
        <tr><th>ID</th><th>图片名</th><th>车牌号</th><th>置信度</th><th>识别时间</th></tr>
        {% for r in records %}
        <tr>
            <td>{{ r[0] }}</td>
            <td>{{ r[1] }}</td>
            <td>{{ r[2] }}</td>
            <td>{{ r[3] }}</td>
            <td>{{ r[4] }}</td>
        </tr>
        {% endfor %}
    </table>
    <p><a href="/">返回检测页面</a></p>
</body>
</html>
"""


# ---------- 7. Flask 路由 ----------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            image_bytes = file.read()
            img_base64, info = detect_plate(image_bytes, image_name=file.filename)
            return render_template_string(HOME_TEMPLATE, image_data=img_base64, info_text=info)
    return render_template_string(HOME_TEMPLATE, image_data=None, info_text=None)


@app.route('/admin', methods=['GET', 'POST'])
def admin():
    message = ""
    if request.method == 'POST':
        plate = request.form.get('plate_number', '').strip()
        owner = request.form.get('owner_name', '').strip()
        is_allowed = 1 if request.form.get('is_allowed') == 'on' else 0
        if plate:
            conn = sqlite3.connect('license_plate.db')
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT OR REPLACE INTO license_plates (plate_number, owner_name, is_allowed) VALUES (?, ?, ?)",
                    (plate, owner, is_allowed))
                conn.commit()
                message = f"添加/更新成功：{plate}"
            except Exception as e:
                message = f"操作失败：{e}"
            finally:
                conn.close()
    # 查询所有车辆
    conn = sqlite3.connect('license_plate.db')
    cursor = conn.cursor()
    cursor.execute("SELECT plate_number, owner_name, is_allowed FROM license_plates ORDER BY id")
    vehicles = cursor.fetchall()
    conn.close()
    return render_template_string(ADMIN_TEMPLATE, vehicles=vehicles, message=message)


@app.route('/delete_vehicle')
def delete_vehicle():
    plate = request.args.get('plate', '')
    if plate:
        conn = sqlite3.connect('license_plate.db')
        cursor = conn.cursor()
        cursor.execute("DELETE FROM license_plates WHERE plate_number=?", (plate,))
        conn.commit()
        conn.close()
    return admin()  # 重定向到管理页


@app.route('/records')
def view_records():
    plate_filter = request.args.get('plate', '')
    conn = sqlite3.connect('license_plate.db')
    cursor = conn.cursor()
    if plate_filter:
        cursor.execute(
            "SELECT id, image_name, plate_text, confidence, timestamp FROM recognition_records WHERE plate_text LIKE ? ORDER BY timestamp DESC",
            (f'%{plate_filter}%',))
    else:
        cursor.execute(
            "SELECT id, image_name, plate_text, confidence, timestamp FROM recognition_records ORDER BY timestamp DESC")
    records = cursor.fetchall()
    conn.close()
    return render_template_string(RECORDS_TEMPLATE, records=records, plate_filter=plate_filter)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)