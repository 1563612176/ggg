import os
import cv2
import torch
import numpy as np
import gradio as gr
import traceback
import warnings

# ================= 解决图表方块字 & 线程崩溃的核心补丁 =================
import matplotlib
matplotlib.use('Agg')  # 强制后台绘图，防止 Gradio 多线程调用时崩溃
import matplotlib.pyplot as plt

# 唤醒 Windows 自带的中文字体 (黑体和微软雅黑)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# =====================================================================

from facenet_pytorch import MTCNN
from models.MAT import MAT

warnings.filterwarnings("ignore")

# ================= 自定义 CSS =================
custom_css = """
#video-container {
    height: 380px !important; 
    border: 2px solid #e5e7eb; 
    border-radius: 10px;
    overflow: hidden;
    display: flex;
    justify-content: center;
    align-items: center;
    background-color: #f9fafb; 
}
#video-container video {
    object-fit: contain !important;
    height: 100% !important;
    width: 100% !important;
}
#analytics-area {
    background-color: #f3f4f6;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
}
"""

print("系统初始化：正在加载检测模型...")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=380, margin=40, keep_all=False, device=DEVICE)

# --- 实例化模型 ---
# 1. 基准模型 (标签逻辑：0=真，1=假)
model_author = MAT(net='efficientnet-b4', feature_layer='b2', attention_layer='b5', M=4)
weights_author = torch.load("pretrained/ff_c23.pth", map_location=DEVICE)
model_author.load_state_dict(weights_author.get('state_dict', weights_author), strict=False)
model_author.to(DEVICE)
model_author.eval()

# 2. 微调模型 (标签逻辑：0=假(fake)，1=真(real))
model_mine = MAT(net='efficientnet-b4', feature_layer='b2', attention_layer='b5', M=4)
weights_mine = torch.load("best_3060_MAT_model.pth", map_location=DEVICE) 
model_mine.load_state_dict(weights_mine.get('state_dict', weights_mine), strict=False)
model_mine.to(DEVICE)
model_mine.eval()

# 模型选项字典
MODELS = {
    "基准预训练模型 (FF++ C23)": model_author,
    "本地微调模型": model_mine,
    "异质双核集成模型 (Ensemble)": "Ensemble"
}
print("模型加载完成，UI 界面已启动。")

# ================= 核心检测逻辑 =================
def detect_deepfake(video_path, model_choice):
    try:
        if video_path is None:
            return "提示：请先上传视频文件。", None, None

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            return "错误：读取视频失败，请检查文件格式。", None, None

        sample_interval = max(1, frame_count // 15)
        fake_probs = []
        tmp_face_path = "gradio_tmp_face.jpg"
        frame_idx = 0

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
                        
                        # 引入高斯滤波降低图像传感器底噪干扰
                        img = cv2.GaussianBlur(img, (3, 3), 0)
                        
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

                        # 执行推理
                        if model_choice == "异质双核集成模型 (Ensemble)":
                            out_author = model_author(img_tensor)
                            out_mine = model_mine(img_tensor)
                            
                            p_author = torch.nn.functional.softmax(out_author, dim=1)[0, 1].item() 
                            p_mine = torch.nn.functional.softmax(out_mine, dim=1)[0, 0].item()   
                            
                            # 加权融合概率
                            prob = (p_author * 0.6) + (p_mine * 0.4)
                            
                        else:
                            active_model = MODELS[model_choice]
                            output = active_model(img_tensor) 
                            
                            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                            
                            if model_choice == "本地微调模型":
                                prob = probabilities[0].item() 
                            else:
                                prob = probabilities[1].item()
                        
                        fake_probs.append(prob)

                frame_idx += 1
        cap.release()

        if os.path.exists(tmp_face_path):
            os.remove(tmp_face_path)

        if not fake_probs:
            return "提示：未能在视频中检测到清晰人脸样本。", {"未知": 1.0}, None

        sorted_probs = sorted(fake_probs)
        # 排除首尾 20% 的极端异常帧，计算均值
        valid_probs = sorted_probs[int(len(sorted_probs)*0.2) : int(len(sorted_probs)*0.8)]
        final_prob = float(np.mean(valid_probs))
        real_prob = 1.0 - final_prob

        # --- 生成检测报告 ---
        result_text = f"分析完成。采用模型：[{model_choice}]。\n共提取并分析 {len(fake_probs)} 帧有效人脸样本。\n"
        if final_prob > 0.5:
            result_text += "判定结果：检测到明显的 Deepfake 伪造特征（疑似合成视频）。"
        else:
            result_text += "判定结果：未见明显伪造痕迹（疑似真实视频）。"

        # --- 绘制时序图表 ---
        fig = plt.figure(figsize=(8, 4))
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='判定阈值 (0.5)')
        plt.plot(range(1, len(fake_probs) + 1), fake_probs, marker='o', color='#3b82f6', linewidth=2, markersize=6)
        plt.fill_between(range(1, len(fake_probs) + 1), fake_probs, alpha=0.1, color='#3b82f6')
        plt.title("逐帧伪造概率时序分析", fontsize=12)
        plt.xlabel("采样帧序号", fontsize=10)
        plt.ylabel("伪造置信度", fontsize=10)
        plt.ylim(-0.05, 1.05)
        plt.legend(loc="upper left")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        return result_text, {"伪造 (Fake)": final_prob, "真实 (Real)": real_prob}, fig

    except Exception as e:
        error_msg = f"系统运行异常：\n\n【错误摘要】：{str(e)}\n\n【堆栈追踪】：\n{traceback.format_exc()}"
        return error_msg, None, None

# ================= UI 界面排版 =================
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("<center><h2>基于多注意力机制的深度伪造检测系统</h2></center>")
    gr.Markdown("<center><h4>毕业设计演示系统 | 支持模型切换与时序分析</h4></center>")

    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Radio(
                choices=["本地微调模型", "基准预训练模型 (FF++ C23)", "异质双核集成模型 (Ensemble)"],
                value="异质双核集成模型 (Ensemble)", 
                label="检测模型选择"
            )
            video_input = gr.Video(label="上传待测视频", elem_id="video-container")
            submit_btn = gr.Button("开始分析", variant="primary")

        with gr.Column(scale=1):
            result_text = gr.Textbox(label="检测报告", lines=3)
            result_label = gr.Label(label="综合概率判定")

    with gr.Row(elem_id="analytics-area"):
        with gr.Column():
            gr.Markdown("### 数据可视化分析")
            prob_plot = gr.Plot(label="时序概率分布折线图")

    submit_btn.click(
        fn=detect_deepfake, 
        inputs=[video_input, model_selector], 
        outputs=[result_text, result_label, prob_plot]
    )

    gr.Markdown("--- \n *系统说明：基于本地受限算力环境开发，采用双模型结构以支撑特征泛化与对比消融实验。*")

if __name__ == "__main__":
    demo.launch(inbrowser=True)