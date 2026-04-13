import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.MAT import MAT
from torch.cuda.amp import autocast, GradScaler # 3060 保命神器：混合精度

# 1. 针对 3060 笔记本的优化配置
DEVICE = torch.device("cuda:0")
BATCH_SIZE = 4 # 显存小，我们小口慢咽
LR = 0.0005    # 微调建议用较小的学习率

# 2. 数据增强（给模型增加难度，防止它死记硬背）
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.RandomHorizontalFlip(), # 随机翻转
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. 加载数据集
train_set = datasets.ImageFolder('D:/train_data/train', transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

# 4. 初始化模型并加载预训练权重（站在巨人肩膀上）
model = MAT(net='efficientnet-b4', feature_layer='b2', attention_layer='b5', M=4)
model.load_state_dict(torch.load("pretrained/ff_c23.pth", map_location=DEVICE), strict=False)
model.to(DEVICE)

# 5. 关键工作量：参数冻结策略
for param in model.parameters():
    param.requires_grad = False # 锁死底层
for param in model.attention_layer.parameters():
    param.requires_grad = True  # 只练注意力层
for param in model._fc.parameters():
    param.requires_grad = True  # 只练分类层

# 6. 开始炼丹
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler() # 混合精度缩放器

print("🚀 炼丹炉已升温，开始训练...")
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        with autocast(): # 开启混合精度，省显存！
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
    print(f"Epoch [{epoch+1}/10] 完成，Loss: {loss.item():.4f}")

# 7. 保存你自己的心血
torch.save(model.state_dict(), "my_finetuned_model.pth")