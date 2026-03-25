import os
import cv2
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

print("============== 自动人脸流水线启动 ==============")

# 1. 绑定你刚刚建好的文件夹路径 (注意路径前面的 r 不要删，防转义)
RAW_DIR = r"D:\Graduation_Project_video\raw_videos"
CROP_DIR = r"D:\Graduation_Project_video\cropped_faces"

# 2. 唤醒显卡并装载 MTCNN
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"正在使用显卡 [{device}] 运行人工智能寻找面部特征...")
# image_size=380 极其关键！直接一步到位切成我们模型需要的尺寸，margin=40 稍微留点边缘
mtcnn = MTCNN(image_size=380, margin=40, keep_all=False, device=device)

# 3. 扫描原始视频
videos = [v for v in os.listdir(RAW_DIR) if v.endswith(('.mp4', '.avi', '.mov'))]
if not videos:
    print("❌ 停停停！raw_videos 文件夹里空空如也。请先往里面放至少一个 mp4 视频！")
    exit()

print(f"✅ 找到 {len(videos)} 个视频，开始疯狂切图...")

# 4. 核心处理循环
for video_name in videos:
    video_path = os.path.join(RAW_DIR, video_name)
    # 为这个视频专门建一个小房间
    vid_name_no_ext = video_name.split('.')[0]
    vid_out_dir = os.path.join(CROP_DIR, vid_name_no_ext)
    os.makedirs(vid_out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n🎥 正在解剖: {video_name} (共计 {frame_count} 帧)")

    frame_idx = 0
    saved_count = 0

    # tqdm 进度条，看着显卡狂飙
    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        # 【抽帧策略】: 视频通常 30帧/秒，每一张都切太浪费空间了！
        # 我们设定每隔 5 帧切一张脸 (也就是1秒截取 6 张脸)
        if frame_idx % 5 == 0:
            # OpenCV 默认是 BGR 格式，而 MTCNN 喜欢 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 设定保存名字，格式如 frame_0000.jpg, frame_0005.jpg
            save_path = os.path.join(vid_out_dir, f"frame_{frame_idx:04d}.jpg")
            
            # MTCNN 极其聪明，给它图片和路径，它会自动抠出脸并保存
            face = mtcnn(frame_rgb, save_path=save_path)
            
            if face is not None:
                saved_count += 1
                
        frame_idx += 1

    cap.release()
    print(f"✅ {video_name} 解剖完成！成功提取了 {saved_count} 张完美对齐的高清人脸。")

print("\n🎉 全部流水线作业完毕！")
print("================================================")