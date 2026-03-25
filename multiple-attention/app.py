import os
import cv2
import torch
import numpy as np
import gradio as gr
from facenet_pytorch import MTCNN
from models.MAT import MAT
import warnings
warnings.filterwarnings("ignore")

# ================= 自定义 CSS =================
# 用于锁定视频播放器框的大小
custom_css = """
#video-container {
    height: 380px !important; /* 设置适合你的固定高度，例如 380px */
    border: 2px solid #e5e7eb; /* 添加边框 */
    border-radius: 10px;
    overflow: hidden;
    display: flex;
    justify-content: center;
    align-items: center;
    background-color: #f9fafb; /* 设置背景色 */
}

/* 关键：强制容器内的视频元素使用 object-fit: contain 并在固定框内按比例缩放 */
#video-container video {
    object-fit: contain !important;
    height: 100% !important;
    width: 100% !important;
}
"""

print("⚙️ 正在预热 AI 引擎，将模型装载进显卡，请稍候...")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载切片机
mtcnn = MTCNN(image_size=380, margin=40, keep_all=False, device=DEVICE)
# 加载大脑
model = MAT(net='efficientnet-b4', feature_layer='b2', attention_layer='b5', M=4)
MODEL_PATH = "pretrained/ff_c23.pth"
weights = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(weights.get('state_dict', weights), strict=False)
model.to(DEVICE)
model.eval()
print("✅ AI 引擎预热完毕！UI 界面即将启动...")

def detect_deepfake(video_path):
    if video_path is None:
        return "⚠️ 请先上传视频文件！", None

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        return "❌ 读取视频失败，请检查文件格式。", None

    # 策略：为了演示时不让老师等太久，固定抽取 15 帧
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
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

                    output = model(img_tensor)
                    prob = torch.nn.functional.softmax(output, dim=1)[0, 1].item()
                    fake_probs.append(prob)

            frame_idx += 1
    cap.release()

    if os.path.exists(tmp_face_path):
        os.remove(tmp_face_path)

    if not fake_probs:
        return "⚠️ 未能在视频中检测到清晰人脸，请换一个视频测试。", {"未知": 1.0}

    # 计算最终得分
    final_prob = np.mean(fake_probs)
    real_prob = 1.0 - final_prob

    # 生成文字报告
    result_text = f"🕵️ 扫描完毕！共提取并分析了 {len(fake_probs)} 帧高清人脸。\n"
    if final_prob > 0.5:
        result_text += f"🚨 终极判定：发现极其明显的 Deepfake 伪造痕迹！"
    else:
        result_text += f"✅ 终极判定：未检测出伪造痕迹，该视频大概率为真实拍摄。"

    # 返回给界面的数据（置信度条形图）
    return result_text, {"🛑 假脸 (Fake)": float(final_prob), "🟢 真脸 (Real)": float(real_prob)}

# ================= UI 界面排版 =================
# 1. 在 gr.Blocks 初始化时传入 custom_css
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("<center><h1>🕵️‍♂️ Deepfake 多注意力机制深伪检测系统</h1></center>")
    gr.Markdown("<center><h3>🎓 毕业设计演示 | 核心架构：EfficientNet-B4 + MAT</h3></center>")

    with gr.Row():
        with gr.Column():
            # 2. 为视频组件添加 elem_id="video-container" 用于绑定 CSS
            video_input = gr.Video(label="📹 拖拽或点击上传待测视频", elem_id="video-container")
            submit_btn = gr.Button("🚀 一键开启 AI 鉴定", variant="primary")

        with gr.Column():
            result_text = gr.Textbox(label="📝 鉴定实验报告", lines=3)
            result_label = gr.Label(label="📊 真假概率置信度")

    # 绑定按钮动作
    submit_btn.click(fn=detect_deepfake, inputs=video_input, outputs=[result_text, result_label])

    gr.Markdown("--- \n *注：本系统由本地 NVIDIA RTX 3060 驱动，采用多注意力机制(MAT)捕捉局部伪造瑕疵。*")

# 启动服务器
if __name__ == "__main__":
    demo.launch(inbrowser=True)