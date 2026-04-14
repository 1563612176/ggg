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
#toolbox-area {
    background-color: #f3f4f6;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
}
"""

print("⚙️ 正在预热 AI 引擎，即将双载入模型，请稍候...")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=380, margin=40, keep_all=False, device=DEVICE)

# --- 实例化两个独立的大脑 ---
# 1. 原作者模型 (标签逻辑：0=真，1=假)
model_author = MAT(net='efficientnet-b4', feature_layer='b2', attention_layer='b5', M=4)
weights_author = torch.load("pretrained/ff_c23.pth", map_location=DEVICE)
model_author.load_state_dict(weights_author.get('state_dict', weights_author), strict=False)
model_author.to(DEVICE)
model_author.eval()

# 2. 你的专属模型 (标签逻辑：ImageFolder 字母排序，0=假(fake)，1=真(real))
model_mine = MAT(net='efficientnet-b4', feature_layer='b2', attention_layer='b5', M=4)
weights_mine = torch.load("best_3060_MAT_model.pth", map_location=DEVICE) 
model_mine.load_state_dict(weights_mine.get('state_dict', weights_mine), strict=False)
model_mine.to(DEVICE)
model_mine.eval()

# --- 找到这一块，增加第三个选项 ---
MODELS = {
    "🌐 原作者基准模型 (FF++ C23)": model_author,
    "🏆 我的 3060 专属微调模型": model_mine,
    "🔥 异质双核集成模式 (Ensemble)": "Ensemble" # 增加这一行
}
print("✅ 双核引擎预热完毕！UI 界面即将启动...")

# ================= 核心检测逻辑 =================
def detect_deepfake(video_path, model_choice):
    try:
        if video_path is None:
            return "⚠️ 请先上传视频文件！", None, None

        active_model = MODELS[model_choice]
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            return "❌ 读取视频失败，请检查文件格式。", None, None

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
                        
                        # 👇 加入“物理外挂”：轻微高斯模糊，抹平摄像头噪点！
                        img = cv2.GaussianBlur(img, (3, 3), 0)
                        
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

                        # 1. 执行推理
                        if model_choice == "🔥 异质双核集成模式 (Ensemble)":
                            # 集成模式：分别跑两个模型，直接计算最终 prob，不需要统一的 output 变量
                            out_author = model_author(img_tensor)
                            out_mine = model_mine(img_tensor)
                            
                            p_author = torch.nn.functional.softmax(out_author, dim=1)[0, 1].item() # 作者逻辑：1是假
                            p_mine = torch.nn.functional.softmax(out_mine, dim=1)[0, 0].item()   # 你的逻辑：0是假
                            
                            # 综合打分 (权重可调)
                            prob = (p_author * 0.6) + (p_mine * 0.4)
                            
                        else:
                            # 单模模式：只跑选中的那一个模型
                            active_model = MODELS[model_choice]
                            output = active_model(img_tensor) # 这里才定义了 output
                            
                            # 将概率计算放在 else 缩进内，确保只有单模模式才执行这一行
                            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                            
                            if "我的 3060" in model_choice:
                                prob = probabilities[0].item() 
                            else:
                                prob = probabilities[1].item()
                        
                        # 2. 统一收集概率结果 (确保这一行在 if/else 外层，且 prob 已被定义)
                        fake_probs.append(prob)

                frame_idx += 1
        cap.release()

        if os.path.exists(tmp_face_path):
            os.remove(tmp_face_path)

        if not fake_probs:
            return "⚠️ 未能在视频中检测到清晰人脸，请换一个视频测试。", {"未知": 1.0}, None

        sorted_probs = sorted(fake_probs)
        # 掐头去尾：去掉最高和最低的 20% 极端帧，只算中间那些稳定帧的平均值
        valid_probs = sorted_probs[int(len(sorted_probs)*0.2) : int(len(sorted_probs)*0.8)]
        final_prob = float(np.mean(valid_probs))
        real_prob = 1.0 - final_prob

        # --- 生成文字报告 ---
        result_text = f"🕵️ [{model_choice.split()[1]}] 扫描完毕！共分析 {len(fake_probs)} 帧高清人脸。\n"
        if final_prob > 0.5:
            result_text += f"🚨 终极判定：发现极其明显的 Deepfake 伪造痕迹！"
        else:
            result_text += f"✅ 终极判定：未检测出伪造痕迹，该视频大概率为真实拍摄。"

        # --- 绘制图表 (现在中文绝对不会乱码了) ---
        fig = plt.figure(figsize=(8, 4))
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='伪造阈值 (0.5)')
        plt.plot(range(1, len(fake_probs) + 1), fake_probs, marker='o', color='#3b82f6', linewidth=2, markersize=6)
        plt.fill_between(range(1, len(fake_probs) + 1), fake_probs, alpha=0.1, color='#3b82f6')
        plt.title("逐帧造假概率分析曲线 (Frame-level Fake Probability)", fontsize=12)
        plt.xlabel("采样帧序号", fontsize=10)
        plt.ylabel("造假置信度", fontsize=10)
        plt.ylim(-0.05, 1.05)
        plt.legend(loc="upper left")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        return result_text, {"🛑 假脸 (Fake)": final_prob, "🟢 真脸 (Real)": real_prob}, fig

    except Exception as e:
        # 兜底捕获：如果再报错，会把具体原因打印在网页上，而不是直接红框报错
        error_msg = f"❌ 代码运行崩溃了！\n\n【简要错误】：{str(e)}\n\n【后台详细报错】：\n{traceback.format_exc()}"
        return error_msg, None, None

# ================= UI 界面排版 =================
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("<center><h1>🕵️‍♂️ Deepfake 多注意力机制深伪检测系统</h1></center>")
    gr.Markdown("<center><h3>🎓 毕业设计演示 | 支持模型一键热切换与时序分析</h3></center>")

    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Radio(
                choices=["🏆 我的 3060 专属微调模型", "🌐 原作者基准模型 (FF++ C23)", "🔥 异质双核集成模式 (Ensemble)"],
                value="🔥 异质双核集成模式 (Ensemble)", # 默认选用集成模式
                label="⚙️ 切换鉴伪引擎 (Model Selector)"
            )
            video_input = gr.Video(label="📹 拖拽或点击上传待测视频", elem_id="video-container")
            submit_btn = gr.Button("🚀 一键开启 AI 鉴定", variant="primary")

        with gr.Column(scale=1):
            result_text = gr.Textbox(label="📝 鉴定实验报告", lines=3)
            result_label = gr.Label(label="📊 综合真假置信度 (Average Probability)")

    with gr.Row(elem_id="toolbox-area"):
        with gr.Column():
            gr.Markdown("### 🛠️ 深度分析工具箱 (Advanced Analytics)")
            gr.Markdown("此处展示检测引擎对视频的深层分析数据。左侧为**时序波动特征**，未来可在此区域横向扩展 Grad-CAM 注意力热力图等可视化组件。")
            
            with gr.Row():
                prob_plot = gr.Plot(label="📉 帧级概率波动折线图")
                placeholder_img = gr.Image(label="🔥 注意力机制热力图 (预留接口)", interactive=False)

    submit_btn.click(
        fn=detect_deepfake, 
        inputs=[video_input, model_selector], 
        outputs=[result_text, result_label, prob_plot]
    )

    gr.Markdown("--- \n *注：本系统由本地 NVIDIA RTX 3060 驱动。双模型热切换技术允许在不重新加载程序的情况下对比微调效果。*")

if __name__ == "__main__":
    demo.launch(inbrowser=True)