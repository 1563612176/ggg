import os
import cv2
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

print("============== 🚀 自动人脸流水线 Pro版 启动 ==============")

# 1. 绑定你的 FF++ 根目录 (指向包含 original, Deepfakes 等文件夹的那个大目录)
# ⚠️ 注意：请务必确保改成你电脑上的实际路径！根据你的截图，应该是下面的路径：
RAW_DIR = r"E:\gravideo\ff-c23\FaceForensics++_C23" 
CROP_DIR = r"E:\gravideo\cropped_faces"

# 2. 唤醒显卡并装载 MTCNN
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"正在使用显卡 [{device}] 运行人工智能寻找面部特征...")
# image_size=380, margin=40 完美适配 MAT 模型的输入需求！
mtcnn = MTCNN(image_size=380, margin=40, keep_all=False, device=device)

# 3. 智能扫描所有视频 (os.walk 支持无限嵌套文件夹，自动穿透！)
video_paths = []
for root, dirs, files in os.walk(RAW_DIR):
    for file in files:
        if file.endswith(('.mp4', '.avi', '.mov')):
            video_paths.append(os.path.join(root, file))

if not video_paths:
    print(f"❌ 停停停！在 {RAW_DIR} 里找不到任何视频。请检查路径！")
    exit()

print(f"✅ 找到 {len(video_paths)} 个视频，开始疯狂切图...")

# 4. 核心处理循环
for video_path in video_paths:
    video_name = os.path.basename(video_path)
    # 动态获取视频所在的父文件夹名字 (比如 'original' 或 'Deepfakes')，按类别存放更清晰
    parent_folder = os.path.basename(os.path.dirname(video_path))
    
    # 为这个视频专门建一个小房间: CROP_DIR / 类别 / 视频名 /
    vid_name_no_ext = video_name.split('.')[0]
    vid_out_dir = os.path.join(CROP_DIR, parent_folder, vid_name_no_ext)
    os.makedirs(vid_out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n🎥 正在解剖: [{parent_folder}] {video_name} (共计 {frame_count} 帧)")

    frame_idx = 0
    saved_count = 0

    # tqdm 进度条，看着显卡狂飙
    for _ in tqdm(range(frame_count), desc=video_name, leave=False):
        ret, frame = cap.read()
        if not ret:
            break

        # 【抽帧策略】: 每隔 5 帧切一张脸
        if frame_idx % 5 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            save_path = os.path.join(vid_out_dir, f"frame_{frame_idx:04d}.jpg")
            
            try:
                # MTCNN 自动寻脸并裁剪
                face = mtcnn(frame_rgb, save_path=save_path)
                if face is not None:
                    saved_count += 1
                    
                    # 🔴 核心保命符：每个视频抽满 25 张立刻停手！防止数据爆炸！
                    if saved_count >= 25:
                        break 
            except Exception as e:
                # 偶尔遇到人脸被严重遮挡或角度太偏，MTCNN会报错，直接跳过不影响大局
                pass
                
        frame_idx += 1

    cap.release()
    print(f"✅ [{parent_folder}] {video_name} 提取了 {saved_count} 张完美对齐的高清人脸。")

print("\n🎉 全部流水线作业完毕！这批“完美饲料”已经准备好送进炼丹炉了！")
print("====================================================================")