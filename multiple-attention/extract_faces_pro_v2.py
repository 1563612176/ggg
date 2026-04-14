import os
import cv2
import torch
import random
from facenet_pytorch import MTCNN
from tqdm import tqdm

print("============== 🚀 自动人脸流水线 (精准采样 80:20 配比版) ==============")

# 1. 绑定路径 (完全贴合你的 E 盘结构)
RAW_DIR = r"E:\gravideo\ff-c23\FaceForensics++_C23" 
CROP_DIR = r"E:\gravideo\cropped_faces"

# 2. 黄金采样配方！精确控制每个文件夹抽取的视频数量
TARGET_COUNTS = {
    'original': 100,        # 真脸 100个视频 * 25张 = 2500张
    'Deepfakes': 20,        # 假脸算法1 * 25 = 500
    'Face2Face': 20,        # 假脸算法2 * 25 = 500
    'FaceShifter': 20,      # 假脸算法3 * 25 = 500
    'FaceSwap': 20,         # 假脸算法4 * 25 = 500
    'NeuralTextures': 20    # 假脸算法5 * 25 = 500
}

# 3. 唤醒显卡并装载 MTCNN
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"正在使用硬件 [{device}] 运行人工智能寻找面部特征...")
mtcnn = MTCNN(image_size=380, margin=40, keep_all=False, device=device)

# 4. 按配方自动随机挑选视频
video_paths = []
print("\n正在按配方随机抽签挑选视频...")

# 为了保证每次实验可复现（比如论文盲审被要求重跑），设定随机种子是一个好习惯
random.seed(42) 

for folder, limit in TARGET_COUNTS.items():
    folder_path = os.path.join(RAW_DIR, folder)
    if not os.path.exists(folder_path):
        print(f"⚠️ 找不到文件夹: {folder}")
        continue
    
    # 找出该文件夹下所有的 mp4/avi 视频
    all_vids = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi'))]
    
    # 防止视频数量不够的保底机制
    actual_limit = min(limit, len(all_vids))
    selected_vids = random.sample(all_vids, actual_limit)
    
    for v in selected_vids:
        video_paths.append(os.path.join(folder_path, v))
    print(f"  - [{folder}] 计划抽取 {limit} 个, 实际抽中 {actual_limit} 个视频")

print(f"\n✅ 抽签完毕！总共锁定 {len(video_paths)} 个目标视频，开始疯狂切图...")

# 5. 核心处理循环
stats = {folder: 0 for folder in TARGET_COUNTS.keys()} # 用于最后统计战果

for video_path in video_paths:
    video_name = os.path.basename(video_path)
    parent_folder = os.path.basename(os.path.dirname(video_path))
    
    vid_name_no_ext = video_name.split('.')[0]
    vid_out_dir = os.path.join(CROP_DIR, parent_folder, vid_name_no_ext)
    os.makedirs(vid_out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ 无法打开视频: {video_name}")
        continue
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    saved_count = 0

    for _ in tqdm(range(frame_count), desc=f"[{parent_folder}] {video_name}", leave=False):
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔 5 帧切一张脸
        if frame_idx % 5 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            save_path = os.path.join(vid_out_dir, f"frame_{frame_idx:04d}.jpg")
            
            try:
                face = mtcnn(frame_rgb, save_path=save_path)
                if face is not None:
                    saved_count += 1
                    stats[parent_folder] += 1 # 计入总统计
                    
                    # 每个视频抽满 25 张立刻停手
                    if saved_count >= 25:
                        break 
            except Exception:
                # 偶尔遇到人脸严重畸变导致 MTCNN 报错，直接跳过
                pass
                
        frame_idx += 1

    cap.release()

# 6. 最终战果汇报
print("\n====================================================================")
print("🎉 全部流水线作业完毕！完美且均衡的数据集已诞生！")
print("📊 最终切片产出统计：")
total_real = stats.get('original', 0)
total_fake = sum(count for folder, count in stats.items() if folder != 'original')
for folder, count in stats.items():
    print(f"  - {folder}: {count} 张")
print(f"\n🟢 总计真脸 (Real): {total_real} 张")
print(f"🔴 总计假脸 (Fake): {total_fake} 张")
print("====================================================================")