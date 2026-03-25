import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
from facenet_pytorch import MTCNN
from models.MAT import MAT
import warnings
warnings.filterwarnings("ignore") # 屏蔽烦人的红字警告

print("============== 🎓 毕设终极图表生成系统启动 ==============")

# --- 1. 配置路径 ---
REAL_DIR = r"D:\Graduation_Project_video\dataset_mini\real"
FAKE_DIR = r"D:\Graduation_Project_video\dataset_mini\fake"
MODEL_PATH = "pretrained/ff_c23.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 2. 唤醒模型 ---
print(f"🚀 核心引擎自检: 正在使用 [{DEVICE}]")
mtcnn = MTCNN(image_size=380, margin=40, keep_all=False, device=DEVICE)
model = MAT(net='efficientnet-b4', feature_layer='b2', attention_layer='b5', M=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE).get('state_dict', torch.load(MODEL_PATH, map_location=DEVICE)), strict=False)
model.to(DEVICE)
model.eval()

# --- 3. 核心探测函数 ---
def scan_dataset(video_dir, label_name):
    videos = [v for v in os.listdir(video_dir) if v.endswith('.mp4')]
    print(f"\n📂 开始扫描 [{label_name}] 阵地，共发现 {len(videos)} 个视频...")
    
    video_scores = []
    for video_name in tqdm(videos, desc=f"化验 {label_name} 视频"):
        video_path = os.path.join(video_dir, video_name)
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = max(1, frame_count // 10) # 提速：每个视频均匀抽 10 帧
        
        fake_probs = []
        frame_idx = 0
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                if frame_idx % sample_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    save_tmp = "tmp_face.jpg"
                    face = mtcnn(frame_rgb, save_path=save_tmp)
                    
                    if face is not None:
                        img = cv2.imread(save_tmp)
                        img = cv2.resize(img, (380, 380))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
                        
                        output = model(img_tensor)
                        prob = torch.nn.functional.softmax(output, dim=1)[0, 1].item()
                        fake_probs.append(prob)
                frame_idx += 1
        cap.release()
        
        if fake_probs:
            video_scores.append(np.mean(fake_probs))
            
    return video_scores

# --- 4. 收集战报 ---
real_scores = scan_dataset(REAL_DIR, "纯真视频 (Real)")
fake_scores = scan_dataset(FAKE_DIR, "伪造视频 (Fake)")

if os.path.exists("tmp_face.jpg"): os.remove("tmp_face.jpg") # 清理临时文件

# --- 5. 组装数据并计算高阶指标 ---
y_true = [0] * len(real_scores) + [1] * len(fake_scores)
y_scores = real_scores + fake_scores
y_pred = [1 if score > 0.5 else 0 for score in y_scores]

acc = accuracy_score(y_true, y_pred)
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

print("\n" + "="*50)
print(f"📈 【最终评估报告】")
print(f"总计检验视频数: {len(y_true)} 个")
print(f"模型综合准确率 (Accuracy): {acc*100:.2f}%")
print(f"模型 AUC 面积值 (越接近1越牛): {roc_auc:.4f}")
print("="*50)

# --- 6. 自动绘制论文图表 ---
print("\n🎨 正在为您绘制毕业论文图表，请稍候...")
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.sans-serif'] = ['SimHei'] # 解决图表中文显示问题
plt.rcParams['axes.unicode_minus'] = False

# 图A：概率分布直方图
plt.figure(figsize=(8, 6), dpi=300)
plt.hist(real_scores, bins=10, alpha=0.7, color='green', label='真实视频 (Real)')
plt.hist(fake_scores, bins=10, alpha=0.7, color='red', label='伪造视频 (Fake)')
plt.axvline(x=0.5, color='black', linestyle='--', label='默认判别阈值 (0.5)')
plt.title('模型对真假视频的判定概率分布 (Probability Distribution)', fontsize=14)
plt.xlabel('判定为"假"的概率得分', fontsize=12)
plt.ylabel('视频数量 (频数)', fontsize=12)
plt.legend(fontsize=12)
plt.savefig('论文图表_概率分布直方图.png', bbox_inches='tight')

# 图B：ROC 曲线
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 曲线 (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (False Positive Rate)', fontsize=12)
plt.ylabel('真阳性率 (True Positive Rate)', fontsize=12)
plt.title('接收者操作特征曲线 (ROC Curve)', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.savefig('论文图表_ROC曲线.png', bbox_inches='tight')

print("✅ 图表绘制完毕！请在代码文件夹查看 [论文图表_概率分布直方图.png] 和 [论文图表_ROC曲线.png]！")