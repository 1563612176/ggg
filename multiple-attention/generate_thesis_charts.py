import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

# ==========================================
# 1. 环境配置：解决中文显示与绘图风格
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.style.use('seaborn-v0_8-paper')           # 使用学术论文风格

def plot_thesis_graphics(data_path='thesis_eval_results.npz'):
    if not os.path.exists(data_path):
        print(f"❌ 错误：找不到数据文件 {data_path}，请先运行数据采集脚本。")
        return

    # 加载数据
    data = np.load(data_path)
    y_true = data['y_true']
    p_base = data['p_base']
    p_local = data['p_local']
    
    # [核心逻辑] 根据论文公式 4-2 计算决策级集成概率
    # Alpha=0.6 代表基准模型权重，0.4 代表本地高敏模型权重
    p_ensemble = 0.6 * p_base + 0.4 * p_local

    # ==========================================
    # 2. 绘制 ROC 曲线对比图 (跨域泛化能力证明)
    # ==========================================
    plt.figure(figsize=(8, 7), dpi=300)
    
    # 计算三组曲线
    models = [
        (y_true, p_base, '基准模型 (EfficientNet-B4)', '#1f77b4', '--'),
        (y_true, p_local, '高敏模型 (MAT-L)', '#ff7f0e', '-.'),
        (y_true, p_ensemble, '决策级异质集成模型 (本文方法)', '#d62728', '-')
    ]

    for y, p, label, color, style in models:
        fpr, tpr, _ = roc_curve(y, p)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, linestyle=style, 
                 label=f'{label} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真阳性率 (True Positive Rate)', fontsize=12)
    plt.title('不同模型在真实野生数据集上的 ROC 曲线对比', fontsize=14, pad=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.savefig('论文图表_ROC曲线.png', bbox_inches='tight')
    print("✅ 已生成：论文图表_ROC曲线.png")

    # ==========================================
    # 3. 绘制概率分布直方图 (防御鲁棒性证明)
    # ==========================================
    plt.figure(figsize=(10, 6), dpi=300)
    
    # 筛选真实视频(label=0)下，两个模型的预测分布
    # 证明：本地模型对真实视频（带噪）容易产生高分误报，而集成模型能将其修正
    real_indices = np.where(y_true == 0)[0]
    real_p_local = p_local[real_indices]
    real_p_ensemble = p_ensemble[real_indices]

    plt.hist(real_p_local, bins=30, alpha=0.5, label='高敏模型 (存在底噪误报)', color='#ff7f0e', edgecolor='white')
    plt.hist(real_p_ensemble, bins=30, alpha=0.7, label='集成模型 (决策平滑后)', color='#2ca02c', edgecolor='white')

    plt.axvline(x=0.5, color='red', linestyle='--', label='分类阈值 (0.5)')
    plt.xlabel('伪造概率预测值 (Score)', fontsize=12)
    plt.ylabel('样本频数 (Frequency)', fontsize=12)
    plt.title('真实场景视频下模型预测概率分布对比', fontsize=14, pad=15)
    plt.legend(loc="upper right")
    
    plt.savefig('论文图表_概率分布直方图.png', bbox_inches='tight')
    print("✅ 已生成：论文图表_概率分布直方图.png")

if __name__ == "__main__":
    plot_thesis_graphics()