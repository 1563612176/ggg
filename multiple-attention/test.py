import os
import cv2
import glob
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from facenet_pytorch import MTCNN
from models.MAT import MAT
import warnings

warnings.filterwarnings("ignore")

print("============== 异质双核集成批量测试流水线 (对齐 app.py 标签逻辑) ==============")

# =====================================================================
# 1. 硬件与路径配置区
# =====================================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 计算引擎自检: 正在使用 [{DEVICE}] 进行张量运算")

# 测试视频目录配置
FAKE_VIDEO_DIR = "datasets/test_samples/fake" 
REAL_VIDEO_DIR = "datasets/test_samples/real"

# 临时切脸保存路径（防内存泄漏，与 app.py 逻辑保持一致）
TMP_FACE_DIR = "gradio_tmp_faces"
os.makedirs(TMP_FACE_DIR, exist_ok=True)

# =====================================================================
# 2. 唤醒模型与特征提取器
# =====================================================================
print("正在加载 MTCNN 人脸提取器...")
mtcnn = MTCNN(image_size=380, margin=40, keep_all=False, device=DEVICE)

print("正在加载 基准预训练模型 (FF++ C23)...")
model_base = MAT(net='efficientnet-b4', feature_layer='b2', attention_layer='b5', M=4)
weights_base = torch.load("pretrained/ff_c23.pth", map_location=DEVICE)
model_base.load_state_dict(weights_base.get('state_dict', weights_base), strict=False)
model_base.to(DEVICE)
model_base.eval()

print("正在加载 本地微调模型 (高敏模型)...")
model_local = MAT(net='efficientnet-b4', feature_layer='b2', attention_layer='b5', M=4)
weights_local = torch.load("best_3060_MAT_model.pth", map_location=DEVICE) 
model_local.load_state_dict(weights_local.get('state_dict', weights_local), strict=False)
model_local.to(DEVICE)
model_local.eval()

# =====================================================================
# 3. 核心工具函数
# =====================================================================
def get_trimmed_mean(probs, trim_ratio=0.2):
    """截断均值聚合 (去除首尾 20% 的极端异常帧，对齐 app.py)"""
    if not probs: return 0.0
    sorted_probs = sorted(probs)
    start_idx = int(len(sorted_probs) * trim_ratio)
    end_idx = int(len(sorted_probs) * (1 - trim_ratio))
    valid_probs = sorted_probs[start_idx:end_idx]
    if not valid_probs: return float(np.mean(probs))
    return float(np.mean(valid_probs))

# =====================================================================
# 4. 批量评测核心管线
# =====================================================================
def run_batch_evaluation(video_folder, true_label, alpha_base=0.6, alpha_local=0.4, report_name="report.csv"):
    if not os.path.exists(video_folder):
        print(f"⚠️ 找不到文件夹: {video_folder}，跳过此阶段测试。")
        return

    video_paths = glob.glob(os.path.join(video_folder, "*.mp4")) + \
                  glob.glob(os.path.join(video_folder, "*.avi"))
    
    total_videos = len(video_paths)
    if total_videos == 0:
        print(f"⚠️ 在 {video_folder} 中未找到视频文件！")
        return

    print(f"\n🎥 开始执行批量测试目录: {video_folder} (共 {total_videos} 个视频)")
    results = []
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    start_time_all = time.time()

    for v_path in tqdm(video_paths, desc="并发推理进度"):
        file_name = os.path.basename(v_path)
        t0 = time.time()
        
        cap = cv2.VideoCapture(v_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            continue
            
        sample_interval = max(1, frame_count // 15)
        base_fake_probs = []
        local_fake_probs = []
        
        frame_idx = 0
        tmp_face_path = os.path.join(TMP_FACE_DIR, f"tmp_{file_name}.jpg")

        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret: break

                if frame_idx % sample_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face = mtcnn(frame_rgb, save_path=tmp_face_path)

                    if face is not None and os.path.exists(tmp_face_path):
                        img = cv2.imread(tmp_face_path)
                        img = cv2.resize(img, (380, 380))
                        
                        # 高斯滤波 (对齐 app.py 抗底噪逻辑)
                        img = cv2.GaussianBlur(img, (3, 3), 0)
                        
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

                        # 双核并发推理
                        out_base = model_base(img_tensor)
                        out_local = model_local(img_tensor)
                        
                        # ========================================================
                        # 严格对齐 app.py 提取【造假置信度(Fake Probability)】逻辑
                        # ========================================================
                        # 基准模型：标签 1 为 Fake
                        p_base_fake = torch.nn.functional.softmax(out_base, dim=1)[0, 1].item() 
                        
                        # 本地微调模型：标签 0 为 Fake
                        p_local_fake = torch.nn.functional.softmax(out_local, dim=1)[0, 0].item()   
                        # ========================================================
                        
                        base_fake_probs.append(p_base_fake)
                        local_fake_probs.append(p_local_fake)
                        
                frame_idx += 1
        cap.release()
        
        if os.path.exists(tmp_face_path):
            os.remove(tmp_face_path)

        # 视频级置信度聚合
        prob_base_final = get_trimmed_mean(base_fake_probs) if base_fake_probs else 0.0
        prob_local_final = get_trimmed_mean(local_fake_probs) if local_fake_probs else 0.0
        
        # 异质双核集成加权公式 (对齐 app.py)
        prob_ensemble = (prob_base_final * alpha_base) + (prob_local_final * alpha_local)
        
        process_time = time.time() - t0
        
        results.append({
            "视频名称": file_name,
            "真实标签": "Fake" if true_label == 1 else "Real",
            "Base造假概率": round(prob_base_final, 4),
            "Local造假概率": round(prob_local_final, 4),
            "集成后造假概率": round(prob_ensemble, 4),
            "Base单核判定": "Fake" if prob_base_final >= 0.5 else "Real",
            "Local单核判定": "Fake" if prob_local_final >= 0.5 else "Real",
            "最终集成判定": "Fake" if prob_ensemble >= 0.5 else "Real",
            "耗时(s)": round(process_time, 3)
        })

    # 统计数据
    total_time = time.time() - start_time_all
    avg_time = total_time / len(results) if results else 0
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

    # 导出 CSV
    df = pd.DataFrame(results)
    df.to_csv(report_name, index=False, encoding='utf-8-sig')
    
    # 打印论文格式战报
    print("\n" + "="*55)
    print(f"📊 批量测试汇总报告 [{report_name}]")
    print("="*55)
    
    if true_label == 1:
        local_intercepts = sum(df["Local单核判定"] == "Fake")
        final_intercepts = sum(df["最终集成判定"] == "Fake")
        print(f"【对应论文 5.4.1 高隐蔽性拦截测试】")
        print(f" - 测试样本总数: {len(results)} 个")
        print(f" - 本地微调单核成功拦截: {local_intercepts} 个 (拦截率 {local_intercepts/len(results):.2%})")
        print(f" - 双核集成模式成功拦截: {final_intercepts} 个 (拦截率 {final_intercepts/len(results):.2%})")
        print(f" - 批次平均输出造假概率 (集成后): {df['集成后造假概率'].mean():.2%}")
    else:
        local_fps = sum(df["Local单核判定"] == "Fake")
        final_fps = sum(df["最终集成判定"] == "Fake")
        print(f"【对应论文 5.4.2 硬件噪声误报压制测试】")
        print(f" - 测试样本总数: {len(results)} 个")
        print(f" - 本地微调单核触发误报: {local_fps} 个")
        print(f" - 双核集成模式压制后误报: {final_fps} 个 (修复了 {local_fps - final_fps} 个误杀)")
        print(f" - 批次平均假阳性概率 (本地单核): {df['Local造假概率'].mean():.2%}")
        print(f" - 批次平均假阳性概率 (双核集成): {df['集成后造假概率'].mean():.2%}")

    print(f"\n【系统吞吐量与计算开销统计】")
    print(f" - 批次总耗时: {total_time:.2f} 秒")
    print(f" - 单视频平均处理耗时: {avg_time:.2f} 秒/个")
    if torch.cuda.is_available():
        print(f" - GPU 峰值显存占用: {peak_memory_mb:.2f} MB")
        
    print(f"\n✅ 详细成绩单已导出至: {report_name}")
    print("="*55)

if __name__ == "__main__":
    os.makedirs(FAKE_VIDEO_DIR, exist_ok=True)
    os.makedirs(REAL_VIDEO_DIR, exist_ok=True)
    
    print(f"提示：请确保假脸(Fake)视频放置于 {FAKE_VIDEO_DIR}")
    print(f"提示：请确保真脸(Real)视频放置于 {REAL_VIDEO_DIR}\n")
    
    # 1. 测 Fake 视频 (验证拦截率)
    run_batch_evaluation(FAKE_VIDEO_DIR, true_label=1, alpha_base=0.6, alpha_local=0.4, report_name="batch_fake_report.csv")
    
    # 2. 测 Real 视频 (验证抗误报率)
    run_batch_evaluation(REAL_VIDEO_DIR, true_label=0, alpha_base=0.6, alpha_local=0.4, report_name="batch_real_report.csv")