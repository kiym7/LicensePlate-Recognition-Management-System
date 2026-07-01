# 车牌检测识别及通行管理系统

基于 YOLOv5 与 PaddleOCR 的车牌检测、识别与通行管理 Web 系统，实现图片上传、自动识别、黑白名单通行判断、识别记录存储与查询等完整功能。

---

## 📌 项目简介

针对当前车牌识别系统功能单一、数据管理薄弱、依赖云端 API 存在隐私风险等问题，设计并实现了一套全本地化的车牌检测识别及通行管理系统。系统采用 YOLOv5s 完成车牌检测，PaddleOCR 完成字符识别，基于 Flask 框架搭建 Web 管理平台，使用 SQLite 数据库存储识别记录和车辆黑白名单信息。经过 30 轮训练，模型在验证集上的 mAP@0.5 达到 91.1%，单张图片处理时间约 0.83 秒。

---

## ✨ 功能特性

| 模块 | 功能 |
|------|------|
| **车牌检测** | YOLOv5s 车牌定位，NMS 后处理，检测框绘制与 base64 输出 |
| **车牌识别** | PaddleOCR 字符识别，支持中文车牌，异常处理（识别失败 / OCR 出错） |
| **通行管理** | 黑白名单增删改查、通行判断（允许 / 禁止 / 未注册）、识别记录自动存储 |
| **历史查询** | 按车牌号模糊搜索历史识别记录 |
| **Web 交互** | Flask 构建主页、管理页、记录页，HTML/CSS 内嵌模板 |

---



## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| 深度学习框架 | PyTorch 2.8.0 |
| 目标检测 | YOLOv5s（本地源码加载） |
| OCR 识别 | PaddleOCR 2.7.3 |
| Web 框架 | Flask 2.0+ |
| 数据库 | SQLite 3 |
| 图像处理 | OpenCV 4.5.5, Pillow |
| 数值计算 | NumPy 1.24.3 |

---

## 📊 数据集与模型

- **数据来源**：Kaggle 公开车牌检测数据集（ANPR YOLO 格式）
- **数据规模**：658 张有效图像（训练集 525 张，验证集 133 张）
- **模型版本**：YOLOv5s
- **训练轮数**：30 轮
- **输入尺寸**：640 × 640
- **批量大小**：8
- **优化器**：SGD（初始学习率 0.01，动量 0.937）
- **性能指标**：
  - mAP@0.5：**0.911**
  - 精确率（Precision）：**0.909**
  - 召回率（Recall）：**0.896**

---

## ⚙️ 环境配置与运行

```bash
# 克隆项目
git clone https://github.com/kiym7/yolov5.git
cd yolov5

# 创建虚拟环境（推荐 Conda）
conda create -n yolov5 python=3.9
conda activate yolov5

# 安装依赖
pip install -r requirements.txt

# 如需 GPU 加速（可选）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 启动 Web 服务
python app_flask.py

##📁 项目结构
├── app_flask.py              # Flask 主程序
├── datasets/                 # 数据集目录
│   ├── images/               # 图片
│   └── labels/               # 标签（YOLO 格式）
├── yolov5/                   # YOLOv5 源码
│   └── runs/train/exp42/weights/best.pt  # 训练好的模型权重
├── license_plate.db          # SQLite 数据库（自动生成）
└── data.yaml                 # 数据集配置文件