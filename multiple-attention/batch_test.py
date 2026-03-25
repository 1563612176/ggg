import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN
from models.MAT import MAT

print("============== 批量 Deepfake 测谎流水线 ==============")

# --- 1. 路径与引擎配置 ---
RAW_REAL_DIR = r"D:\Graduation_Project_video\dataset_mini\real"
CROP_REAL_DIR = r"D:\Graduation_Project_video\cropped_faces\real"
MODEL_PATH = "pretrained/ff_c23.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.makedirs(CROP_REAL_DIR, exist_ok=True)
print(f"🚀 计算引擎自检: 正在使用 [{DEVICE}] 火力全开！")

# --- 2. 唤醒两大核心 AI ---
print("正在加载 MTCNN 切片机...")
mtcnn = MTCNN(image_size=380, margin=40, keep_all=False, device=DEVICE)

print(f"正在加载 MAT 满级大脑...")
model = MAT(net='efficientnet-b4', feature_layer='b2', attention_layer='b5', M=4)
weights = torch.load(MODEL_PATH, map_location=DEVICE)
if 'state_dict' in weights:
    model.load_state_dict(weights['state_dict'], strict=False)
else:
    model.load_state_dict(weights, strict=False)
model.to(DEVICE)
model.eval()

# --- 3. 扫描视频库 ---
videos = [v for v in os.listdir(RAW_REAL_DIR) if v.endswith('.mp4')]
print(f"\n✅ 成功接驳数据库，找到 {len(videos)} 个测试视频！")

# 记录每个视频的最终得分
video_results = {}

# --- 4. 开始工业化流水线 ---
for video_name in videos:
    video_path = os.path.join(RAW_REAL_DIR, video_name)
    vid_name_no_ext = video_name.split('.')[0]
    vid_crop_dir = os.path.join(CROP_REAL_DIR, vid_name_no_ext)
    os.makedirs(vid_crop_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 策略：为了提速，我们每个视频均匀抽取 15 帧进行化验
    sample_interval = max(1, frame_count // 15)
    
    print(f"\n🎥 正在检验: {video_name} ...")
    fake_probs = []
    
    frame_idx = 0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret: break
                
            if frame_idx % sample_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                save_path = os.path.join(vid_crop_dir, f"frame_{frame_idx:04d}.jpg")
                
                # 1. 切脸
                face = mtcnn(frame_rgb, save_path=save_path)
                
                if face is not None:
                    # 2. 读取切好的脸并预处理
                    img = cv2.imread(save_path)
                    img = cv2.resize(img, (380, 380))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
                    
                    # 3. 测谎仪推理
                    output = model(img_tensor)
                    prob = torch.nn.functional.softmax(output, dim=1)[0, 1].item()
                    fake_probs.append(prob)
            
            frame_idx += 1
            
    cap.release()
    
    # 汇总这个视频的得分
    if fake_probs:
        final_prob = np.mean(fake_probs)
        video_results[video_name] = final_prob
        print(f"   => 假脸概率: {final_prob*100:.2f}%")
    else:
        print(f"   => ⚠️ 未能在视频中检测到清晰人脸")

# --- 5. 打印最终成绩单 ---
print("\n" + "="*50)
print("📊 【真视频测试批次 - 最终成绩单】")
print("="*50)
for v_name, prob in video_results.items():
    status = "✅ 判定为真" if prob < 0.5 else "🚩 误报为假"
    print(f"{v_name:<20} | 假脸概率: {prob*100:>6.2f}% | {status}")
print("="*50)