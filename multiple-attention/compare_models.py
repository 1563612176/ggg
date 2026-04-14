import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from facenet_pytorch import MTCNN
import warnings

# 导入模型架构
try:
    from models.MAT import MAT
except ImportError:
    print("⚠️ 运行失败：请确保在 multiple-attention 根目录下运行此脚本！")

warnings.filterwarnings("ignore")

print("===================================================================")
print("⚔️  Deepfake 异质双核集成判定测试流水线 V2.0 启动")
print("===================================================================\n")

# --- 1. 配置路径 ---
TEST_DIR = r"E:\gravideo\wild_test_videos"  # 你存放待测视频的目录
REPORT_PATH = "ensemble_comparison_report.csv"

# --- 2. 预热双核引擎 ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"⚙️  正在唤醒 [{DEVICE}] 显卡并装载双模型...")

mtcnn = MTCNN(image_size=380, margin=40, keep_all=False, device=DEVICE)

# A. 原作者模型 (逻辑：1代表假)
model_author = MAT(net='efficientnet-b4', feature_layer='b2', attention_layer='b5', M=4)
weights_a = torch.load("pretrained/ff_c23.pth", map_location=DEVICE)
model_author.load_state_dict(weights_a.get('state_dict', weights_a), strict=False)
model_author.to(DEVICE)
model_author.eval()

# B. 你的 3060 专属模型 (逻辑：0代表假)
model_mine = MAT(net='efficientnet-b4', feature_layer='b2', attention_layer='b5', M=4)
weights_m = torch.load("best_3060_MAT_model.pth", map_location=DEVICE)
model_mine.load_state_dict(weights_m.get('state_dict', weights_m), strict=False)
model_mine.to(DEVICE)
model_mine.eval()

print("✅ 双核引擎加载成功，物理降噪补丁(GaussianBlur)已就绪。\n")

# --- 3. 扫描视频文件 ---
if not os.path.exists(TEST_DIR):
    print(f"❌ 错误：找不到路径 {TEST_DIR}")
    exit()

video_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
if not video_files:
    print("⚠️ 文件夹内没有可识别的视频文件。")
    exit()

# --- 4. 批量执行 ---
results_list = []

for video_name in tqdm(video_files, desc="全自动审计进度"):
    video_path = os.path.join(TEST_DIR, video_name)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count == 0: continue

    sample_interval = max(1, frame_count // 15)
    
    # 存储每一帧的概率
    author_frame_probs = []
    mine_frame_probs = []
    
    tmp_face_path = "batch_temp.jpg"
    frame_idx = 0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret: break

            if frame_idx % sample_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # MTCNN 提取
                face = mtcnn(frame_rgb, save_path=tmp_face_path)

                if face is not None and os.path.exists(tmp_face_path):
                    img = cv2.imread(tmp_face_path)
                    img = cv2.resize(img, (380, 380))
                    
                    # --- 物理外挂：高斯降噪，平滑摄像头噪点 ---
                    img = cv2.GaussianBlur(img, (3, 3), 0)
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

                    # 推理
                    out_a = model_author(img_tensor)
                    out_m = model_mine(img_tensor)
                    
                    # 概率映射转换
                    p_a = torch.nn.functional.softmax(out_a, dim=1)[0, 1].item() # 1是假
                    p_m = torch.nn.functional.softmax(out_m, dim=1)[0, 0].item() # 0是假
                    
                    author_frame_probs.append(p_a)
                    mine_frame_probs.append(p_m)

            frame_idx += 1
    cap.release()
    if os.path.exists(tmp_face_path): os.remove(tmp_face_path)

    # --- 数据聚合与集成逻辑 ---
    if author_frame_probs:
        avg_p_author = np.mean(author_frame_probs)
        avg_p_mine = np.mean(mine_frame_probs)
        
        # 👑 集成决策逻辑：0.6 作者 + 0.4 你
        # 这个比例可以根据你的测试集表现进行微调
        avg_p_ensemble = (avg_p_author * 0.6) + (avg_p_mine * 0.4)
        
        v_author = "Fake" if avg_p_author > 0.5 else "Real"
        v_mine = "Fake" if avg_p_mine > 0.5 else "Real"
        v_ensemble = "Fake" if avg_p_ensemble > 0.5 else "Real"
        
        status = "🤝 一致" if v_author == v_mine else "🔥 集成修正"
    else:
        avg_p_author = avg_p_mine = avg_p_ensemble = 0
        v_author = v_mine = v_ensemble = "无脸"
        status = "跳过"

    results_list.append({
        "视频": video_name,
        "作者模型(Prob)": f"{avg_p_author*100:.1f}%",
        "我的模型(Prob)": f"{avg_p_mine*100:.1f}%",
        "集成模型(Prob)": f"{avg_p_ensemble*100:.1f}%",
        "最终判定": v_ensemble,
        "状态": status
    })

# --- 5. 导出报表 ---
df = pd.DataFrame(results_list)
print("\n" + df.to_string(index=False) + "\n")
df.to_csv(REPORT_PATH, index=False, encoding='utf-8-sig')
print(f"🎉 报表已生成：{REPORT_PATH}")