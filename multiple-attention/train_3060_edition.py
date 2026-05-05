import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt  # [新增] 导入绘图库

try:
    from models.MAT import MAT
except ImportError:
    print("⚠️ 请确保你的终端目前正处于 multiple-attention 文件夹下，且存在 models 文件夹！")

# --- 1. 核心超参数配置 ---
DATA_ROOT = r'E:\gravideo\final_dataset'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PHYSICAL_BATCH_SIZE = 4 
ACCUMULATION_STEPS = 4  
EPOCHS = 20
LR = 0.0005

# ==========================================
# 🛠️ 核心 Bug 修复 1：定制人脸真伪专用的 ImageFolder
# ==========================================
class DeepfakeImageFolder(datasets.ImageFolder):
    def find_classes(self, directory: str):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"找不到类别文件夹: {directory}")

        class_to_idx = {}
        for cls_name in classes:
            if 'real' in cls_name.lower() or 'origin' in cls_name.lower():
                class_to_idx[cls_name] = 0
            else:
                class_to_idx[cls_name] = 1
                
        print(f"📁 文件夹 [{os.path.basename(directory)}] 标签映射结果: {class_to_idx}")
        return classes, class_to_idx


# ==========================================
# 🔴 Windows 保命锁：把核心流程关进 main 里
# ==========================================
if __name__ == '__main__':
    print("🔥 RTX 3060 极限炼丹炉启动 (收敛图增强版)...")

    # --- 2. 数据加载与增强 ---
    train_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
    ])

    val_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
    ])

    train_set = DeepfakeImageFolder(os.path.join(DATA_ROOT, 'train'), transform=train_transform)
    val_set = DeepfakeImageFolder(os.path.join(DATA_ROOT, 'val'), transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=PHYSICAL_BATCH_SIZE, shuffle=False, num_workers=2)

    # --- 3. 初始化模型 ---
    model = MAT(net='efficientnet-b4', feature_layer='b2', attention_layer='b5', M=4)

    weight_path = "pretrained/ff_c23.pth"
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE), strict=False)

    model = model.to(DEVICE)

    for name, param in model.named_parameters():
        param.requires_grad = False
        if "attention_layer" in name or "_fc" in name or "blocks.6" in name:
            param.requires_grad = True

    # --- 4. 损失函数与优化器 ---
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() 

    best_val_acc = 0.0
    train_losses = []  # [新增] 用于存储每个 epoch 的平均损失

    # --- 5. 核心训练循环 ---
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad()
        
        print(f"\n--- Epoch [{epoch+1}/{EPOCHS}] ---")
        train_bar = tqdm(train_loader, desc="Training")
        
        for i, (images, labels) in enumerate(train_bar):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / ACCUMULATION_STEPS 
            
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            running_loss += loss.item() * ACCUMULATION_STEPS
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            train_bar.set_postfix({'Loss': f"{loss.item() * ACCUMULATION_STEPS:.4f}", 'Acc': f"{100.*correct/total:.2f}%"})

        # [新增] 计算并记录当前 epoch 的平均训练损失
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # --- 6. 验证循环 ---
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with autocast():
                    outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_acc = 100. * val_correct / val_total
        print(f"📊 [Epoch {epoch+1} 总结] Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_3060_MAT_model.pth")
            print(f"🏆 保存最佳权重 -> best_3060_MAT_model.pth")

    print("\n🎉 训练结束！正在生成损失收敛曲线...")

    # [新增] 绘制并保存损失收敛曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_losses, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.xticks(range(1, EPOCHS + 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 保存图片，不设标题
    plot_path = "fine_tuning_loss_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📈 损失收敛曲线图已保存至: {plot_path}")

    print("\n🎉 全部工作已完成！")