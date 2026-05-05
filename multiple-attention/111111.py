import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, f1_score

#画第四章的图！！！


# 设置绘图风格，适合论文插入
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def calculate_metrics(y_true, y_prob, threshold=0.5):
    """计算核心评估指标: Accuracy, Recall, FPR, Precision, F1"""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred)
    
    return accuracy, recall, fpr, precision, f1

def grid_search_alpha(y_true, p_base, p_local):
    """实验2：动态权重寻优 (Grid Search)"""
    alphas = np.arange(0.1, 1.0, 0.1)
    recalls = []
    fprs = []
    accuracies = []

    for alpha in alphas:
        # P_final = α * P_base + (1-α) * P_local
        p_final = alpha * p_base + (1 - alpha) * p_local
        acc, rec, fpr, _, _ = calculate_metrics(y_true, p_final)
        
        recalls.append(rec)
        fprs.append(fpr)
        accuracies.append(acc)

    # 绘制寻优折线图
    plt.figure(figsize=(8, 6))
    plt.plot(alphas, recalls, marker='o', label='召回率 (Recall) - 伪造拦截', color='red')
    plt.plot(alphas, fprs, marker='s', label='假阳性率 (FPR) - 野生环境误报', color='blue')
    plt.plot(alphas, accuracies, marker='^', label='综合准确率 (Accuracy)', color='green', linestyle='--')
    
    plt.axvline(x=0.6, color='gray', linestyle=':', label='最优权重 α=0.6')
    plt.title('决策级集成权重 α 动态寻优过程')
    plt.xlabel('基准模型权重 α')
    plt.ylabel('性能指标 (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('grid_search_alpha.png', dpi=300)
    plt.show()

def plot_roc_curves(y_true, p_base, p_local, p_ensemble):
    """实验3：绘制异质双核与单模型的 ROC 曲线对比"""
    plt.figure(figsize=(8, 6))
    
    # 计算并绘制基准模型 ROC
    fpr_b, tpr_b, _ = roc_curve(y_true, p_base)
    auc_b = auc(fpr_b, tpr_b)
    plt.plot(fpr_b, tpr_b, label=f'基准预训练模型 (AUC = {auc_b:.3f})', linestyle='--')
    
    # 计算并绘制本地微调模型 ROC
    fpr_l, tpr_l, _ = roc_curve(y_true, p_local)
    auc_l = auc(fpr_l, tpr_l)
    plt.plot(fpr_l, tpr_l, label=f'本地高敏微调模型 (AUC = {auc_l:.3f})', linestyle='-.')
    
    # 计算并绘制集成模型 ROC
    fpr_e, tpr_e, _ = roc_curve(y_true, p_ensemble)
    auc_e = auc(fpr_e, tpr_e)
    plt.plot(fpr_e, tpr_e, label=f'异质双核集成模型 (AUC = {auc_e:.3f})', color='red', linewidth=2)
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (False Positive Rate)')
    plt.ylabel('真正例率 (True Positive Rate)')
    plt.title('深度伪造检测跨域性能 ROC 曲线对比')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('roc_comparison.png', dpi=300)
    plt.show()

def main():
    # ==========================================
    # 模拟数据准备 (请替换为你真实的推理输出)
    # y_true: 0代表真实视频(负例), 1代表伪造视频(正例)
    # ==========================================
    np.random.seed(42)
    num_fake = 60 # 论文中提及60个伪造样本
    num_real = 34 # 论文中提及34个真实野生样本
    y_true = np.array([1]*num_fake + [0]*num_real)
    
    # 模拟基准模型输出: 漏报较高，但对真实视频不易误报
    p_base_fake = np.random.normal(loc=0.6, scale=0.2, size=num_fake)
    p_base_real = np.random.normal(loc=0.2, scale=0.15, size=num_real)
    p_base = np.clip(np.concatenate([p_base_fake, p_base_real]), 0, 1)
    
    # 模拟本地模型输出: 拦截能力极强，但受ISP噪声影响对真实样本产生高误报
    p_local_fake = np.random.normal(loc=0.9, scale=0.1, size=num_fake)
    p_local_real = np.random.normal(loc=0.65, scale=0.2, size=num_real) 
    p_local = np.clip(np.concatenate([p_local_fake, p_local_real]), 0, 1)

    # 实验1：打印表 4-1 的对比数据
    print("=== 表 4-1 跨域性能对比实验 ===")
    
    # 最优权重 α=0.6
    alpha_optimal = 0.6
    p_ensemble = alpha_optimal * p_base + (1 - alpha_optimal) * p_local
    
    models = {'基准模型': p_base, '本地模型': p_local, '双核集成': p_ensemble}
    for name, prob in models.items():
        acc, rec, fpr, _, _ = calculate_metrics(y_true, prob)
        print(f"{name} -> 准确率: {acc*100:.2f}%, 召回率: {rec*100:.2f}%, 误报率: {fpr*100:.2f}%")

    # 实验2：执行动态权重寻优
    print("\n=== 执行动态权重寻优 ===")
    grid_search_alpha(y_true, p_base, p_local)
    
    # 实验3：绘制 ROC 曲线
    print("\n=== 生成 ROC 曲线 ===")
    plot_roc_curves(y_true, p_base, p_local, p_ensemble)

if __name__ == "__main__":
    main()