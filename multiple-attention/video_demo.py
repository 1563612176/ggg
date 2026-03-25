import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from models.MAT import MAT

print("============== 视频级 Deepfake 终极鉴定 ==============")

# --- 1. 配置路径 ---
# 注意：这里假设你的视频名字叫 1.mp4，所以切出来的文件夹叫 "1"
# 如果你的视频叫别的名字，请把这里的 "1" 改成对应的文件夹名！
VIDEO_FACES_DIR = r"D:\Graduation_Project_video\cropped_faces\1" 
MODEL_PATH = "pretrained/ff_c23.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"计算引擎自检: 正在使用 [{DEVICE}] 进行高维张量运算...")

# --- 2. 唤醒满级大脑 ---
print(f"正在加载架构并读取权重: {MODEL_PATH}...")
model = MAT(net='efficientnet-b4', feature_layer='b2', attention_layer='b5', M=4)
weights = torch.load(MODEL_PATH, map_location=DEVICE)
if 'state_dict' in weights:
    model.load_state_dict(weights['state_dict'], strict=False)
else:
    model.load_state_dict(weights, strict=False)
model.to(DEVICE)
model.eval()

# --- 3. 批量读取并鉴定 ---
face_images = [f for f in os.listdir(VIDEO_FACES_DIR) if f.endswith('.jpg')]
if not face_images:
    print("❌ 找不到人脸图片，请检查文件夹路径是否正确！")
    exit()

print(f"✅ 成功接驳数据库，找到 {len(face_images)} 张人脸切片，开始逐帧鉴定...")

fake_probs = []

with torch.no_grad():
    for img_name in tqdm(face_images):
        img_path = os.path.join(VIDEO_FACES_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None: continue

        # 预处理对齐 (虽然已经是 380x380，但过一遍标准流程更严谨)
        img = cv2.resize(img, (380, 380))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

        # 深度学习推理
        output = model(img)
        prob = torch.nn.functional.softmax(output, dim=1)[0, 1].item()
        fake_probs.append(prob)

# --- 4. 汇总出具报告 ---
final_prob = np.mean(fake_probs)

print("\n" + "="*50)
print(f"📊 【1.mp4 视频最终鉴定报告】")
print(f"总计检测帧数: {len(fake_probs)} 帧")
print(f"单帧最高造假概率: {max(fake_probs)*100:.2f}%")
print(f"单帧最低造假概率: {min(fake_probs)*100:.2f}%")
print(f"--------------------------------------------------")
print(f"🚨 视频综合 Deepfake (假脸) 概率: {final_prob * 100:.2f}%")

if final_prob > 0.5:
    print("🚩 权威结论：该视频存在极其明显的 AI 换脸/伪造痕迹！")
else:
    print("✅ 权威结论：该视频未检测出伪造痕迹，判定为真实拍摄。")
print("="*50)