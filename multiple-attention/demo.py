import torch
import cv2
import numpy as np
from models.MAT import MAT

# --- 1. 配置区域 ---
IMG_PATH = "test_face.jpg"  # 你的照片路径
MODEL_PATH = "pretrained/ff_c23.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("============== 真实图片验证程序 ==============")

# --- 2. 组装模型 ---
print(f"正在加载架构并读取权重: {MODEL_PATH}...")
model = MAT(net='efficientnet-b4', feature_layer='b2', attention_layer='b5', M=4)
weights = torch.load(MODEL_PATH, map_location=DEVICE)
if 'state_dict' in weights:
    model.load_state_dict(weights['state_dict'], strict=False)
else:
    model.load_state_dict(weights, strict=False)
model.to(DEVICE)
model.eval()

# --- 3. 图像预处理 (核心步骤) ---
print(f"正在预处理图片: {IMG_PATH}...")
img = cv2.imread(IMG_PATH)
if img is None:
    print(f"❌ 报错: 找不到图片 {IMG_PATH}，请检查文件名！")
    exit()

# 统一缩放到 380x380 (论文要求的尺寸)
img = cv2.resize(img, (380, 380))
# BGR 转 RGB 并归一化到 [0, 1]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
# 转换为 Tensor 格式 (Channels, Height, Width)
img = torch.from_numpy(img).permute(2, 0, 1).float()
# 增加一个 Batch 维度 (1, 3, 380, 380)
img = img.unsqueeze(0).to(DEVICE)

# --- 4. 开始推理 ---
print("正在通过多注意力机制进行鉴定...")
with torch.no_grad():
    output = model(img)
    prob = torch.nn.functional.softmax(output, dim=1)[0, 1].item()

print("\n" + "="*40)
print(f"🔍 鉴定结果：")
print(f"这张人脸是 Deepfake (假脸) 的可能性为: {prob * 100:.2f}%")
if prob > 0.5:
    print("🚩 结论：检测到明显的伪造痕迹！")
else:
    print("✅ 结论：该人脸被判定为真实。")
print("="*40)