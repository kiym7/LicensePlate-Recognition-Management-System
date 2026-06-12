import torch
from PIL import Image

# 请修改为你的模型实际路径
MODEL_PATH = r"D:\Bishe\yolov5\runs\train\exp3\weights\best.pt"
# 请修改为你要测试的图片路径（可以是任何图片）
TEST_IMAGE = r"D:\Bishe\train_sample.jpg"   # 换成你的图片路径

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 加载模型
print("加载模型中...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=False, trust_repo=True)
model.to(device)
model.conf = 0.25   # 低阈值
print(f"模型类别: {model.names}")
print(f"置信度阈值: {model.conf}")

# 加载图片
img = Image.open(TEST_IMAGE).convert('RGB')
print(f"图片尺寸: {img.size}")

# 推理
results = model(img)
detections = results.pandas().xyxy[0]
print(f"检测到 {len(detections)} 个目标")
if len(detections) > 0:
    print(detections[['confidence', 'xmin', 'ymin', 'xmax', 'ymax', 'name']])
else:
    print("未检测到任何目标。可能原因：1) 模型不是车牌检测模型；2) 图片中没有车牌；3) 阈值仍需降低。")

# 可选：保存带标注的结果图
results.save(save_dir='./test_output')
print("带标注的结果已保存到 ./test_output")