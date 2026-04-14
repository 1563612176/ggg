import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

try:
    from models.MAT import MAT
except ImportError:
    print("⚠️ 请确保你的终端目前正处于 multiple-attention 文件夹下，且存在 models 文件夹！")

# --- 1. 核心超参数配置 (全局变量可以放外面) ---
DATA_ROOT = r'E:\gravideo\final_dataset'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PHYSICAL_BATCH_SIZE = 4 
ACCUMULATION_STEPS = 4  
EPOCHS = 20
LR = 0.0005

# ==========================================
# 🔴 Windows 保命锁：把核心流程关进 main 里
# ==========================================
if __name__ == '__main__':
    print("🔥 RTX 3060 极限炼丹炉启动 (Windows 适配版)...")

    # --- 2. 数据加载与增强 ---
    train_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.05, contrast=0.05),#从0.15降到0.05，使模型保留一定的抗干扰能力，但不至于让模型过度敏感
    # 👇 新增：随机给 30% 的图片加上高斯模糊，模拟低端摄像头失焦
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
    # 👇 新增：随机降低锐度，模拟视频高强度压缩带来的涂抹感
    transforms.RandomAdjustSharpness(sharpness_factor=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

    val_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder(os.path.join(DATA_ROOT, 'train'), transform=train_transform)
    val_set = datasets.ImageFolder(os.path.join(DATA_ROOT, 'val'), transform=val_transform)

    # 这里的 num_workers=2 在 Windows 下必须被 if __name__ == '__main__' 保护！
    train_loader = DataLoader(train_set, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=PHYSICAL_BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"📦 数据加载完毕: 训练集 {len(train_set)} 张 | 验证集 {len(val_set)} 张")

    # --- 3. 初始化模型 ---
    model = MAT(net='efficientnet-b4', feature_layer='b2', attention_layer='b5', M=4)

    weight_path = "pretrained/ff_c23.pth"
    if os.path.exists(weight_path):
        print(f"🔄 正在加载预训练权重: {weight_path}")
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE), strict=False)
    else:
        print("⚠️ 未找到预训练权重，模型将从头开始随机初始化 (不推荐)！")

    model = model.to(DEVICE)

    # 显存保命策略：解冻高层和注意力机制
    for name, param in model.named_parameters():
        param.requires_grad = False
        if "attention_layer" in name or "_fc" in name or "blocks.6" in name:
            param.requires_grad = True

    # --- 4. 损失函数与优化器 ---
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() 

    best_val_acc = 0.0

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

        # --- 6. 验证循环 ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_acc = 100. * val_correct / val_total
        print(f"📊 [Epoch {epoch+1} 总结] Train Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_3060_MAT_model.pth")
            print(f"🏆 新纪录！已保存最佳模型权重 -> best_3060_MAT_model.pth")

    print("\n🎉 炼丹彻底结束！你的专属模型已出炉！")