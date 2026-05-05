import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN
import warnings

# 1. 确保根目录在系统路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 2. 导入现成的 MAT 模型
from models.MAT import MAT

warnings.filterwarnings("ignore")

# ==========================================
# 3. 核心配置 (根据 compare_models.py 提取)
# ==========================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ALPHA = 0.6  # 论文中的集成权重
GAMMA = 0.1  # 截断均值比例

# --- 待测数据集目录 (请根据实际情况修改) ---
TEST_DIRS = {
    "fake": r"E:\gravideo\wild_test_videos\fake", # 伪造视频存放处
    "real": r"E:\gravideo\wild_test_videos\real"  # 真实视频存放处
}

# --- 权重文件路径 ---
WEIGHTS_BASE = "pretrained/ff_c23.pth"        # 作者基准模型
WEIGHTS_LOCAL = "best_3060_MAT_model.pth"     # 你的3060模型

# ==========================================
# 4. 论文公式实现：截断均值
# ==========================================
def trimmed_mean(probs, gamma=0.1):
    """直接在脚本内定义，不依赖 utils.py，对应论文公式(3-4)"""
    if len(probs) == 0: return 0.5
    probs_sorted = np.sort(probs)
    m = int(len(probs) * gamma)
    if m > 0: probs_sorted = probs_sorted[m:-m]
    return np.mean(probs_sorted)

# ==========================================
# 5. 实验主循环
# ==========================================
def main():
    print("=====================================================")
    print("🚀 毕业设计 - 异质双核跨域性能评估与数据采集流水线")
    print("=====================================================\n")

    # [加载模型] 完全采用 compare_models.py 的参数: feature='b2', attention='b5', M=4
    print(f"⚙️ 正在加载双核模型至 {DEVICE} ...")
    
    # 1. 基准模型
    model_base = MAT(net='efficientnet-b4', feature_layer='b2', attention_layer='b5', M=4)
    if os.path.exists(WEIGHTS_BASE):
        weights_a = torch.load(WEIGHTS_BASE, map_location=DEVICE)
        model_base.load_state_dict(weights_a.get('state_dict', weights_a), strict=False)
    model_base.to(DEVICE).eval()

    # 2. 你的本地高敏模型
    model_local = MAT(net='efficientnet-b4', feature_layer='b2', attention_layer='b5', M=4)
    if os.path.exists(WEIGHTS_LOCAL):
        weights_m = torch.load(WEIGHTS_LOCAL, map_location=DEVICE)
        model_local.load_state_dict(weights_m.get('state_dict', weights_m), strict=False)
    model_local.to(DEVICE).eval()

    # [加载 MTCNN]
    mtcnn = MTCNN(image_size=380, margin=40, keep_all=False, device=DEVICE)

    results = {"y_true": [], "p_base": [], "p_local": []}
    tmp_face_path = "temp_face_eval.jpg"

    # [遍历目录]
    for label, (cat, path) in enumerate(TEST_DIRS.items()):
        # 约定：字典遍历 fake 时 label=0 (假)，real 时 label=1 (真)
        # 但论文习惯中通常正例(1)是伪造，负例(0)是真实。我们在保存前修正。
        actual_label = 1 if cat == "fake" else 0
        
        if not os.path.exists(path):
            print(f"⚠️ 警告: 找不到路径 {path}，将跳过。")
            continue
            
        vids = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.mp4', '.avi'))]
        print(f"\n📂 开始处理 [{cat}] 视频，共 {len(vids)} 个...")
        
        for v_path in tqdm(vids):
            cap = cv2.VideoCapture(v_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0: continue
            
            # 自适应抽帧：约 15-25 帧
            sample_interval = max(1, frame_count // 15)
            
            p_base_list, p_local_list = [], []
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                if frame_idx % sample_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face = mtcnn(frame_rgb, save_path=tmp_face_path)
                    
                    if face is not None and os.path.exists(tmp_face_path):
                        img = cv2.imread(tmp_face_path)
                        img = cv2.resize(img, (380, 380))
                        
                        # [极其关键：复用 compare_models.py 的物理降噪外挂]
                        img = cv2.GaussianBlur(img, (3, 3), 0)
                        
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
                        
                        with torch.no_grad():
                            out_base = model_base(img_tensor)
                            out_local = model_local(img_tensor)
                            
                            # [极其关键：标签反转修复] 
                            # 基准模型 1 是假
                            pb = torch.nn.functional.softmax(out_base, dim=1)[0, 1].item() 
                            # 你的模型 0 是假
                            pl = torch.nn.functional.softmax(out_local, dim=1)[0, 0].item() 
                            
                            p_base_list.append(pb)
                            p_local_list.append(pl)
                            
                frame_idx += 1
            cap.release()
            
            if len(p_base_list) > 0:
                results["y_true"].append(actual_label)
                results["p_base"].append(trimmed_mean(p_base_list, GAMMA))
                results["p_local"].append(trimmed_mean(p_local_list, GAMMA))

    if os.path.exists(tmp_face_path):
        os.remove(tmp_face_path)

    # 6. 保存供画图的 NPZ 数据
    save_path = 'thesis_eval_results.npz'
    np.savez(save_path, 
             y_true=np.array(results["y_true"]), 
             p_base=np.array(results["p_base"]), 
             p_local=np.array(results["p_local"]))
             
    print(f"\n🎉 实验数据采集完成！已保存至: {save_path}")
    print("👉 下一步：直接运行项目里的 generate_thesis_charts.py 生成论文配图！")

if __name__ == "__main__":
    main()