import os

print("===============================================================")
print("🛡️ 毕业设计数据集终极审计程序 (Data Audit Report)")
print("===============================================================\n")

# --- 1. 绑定终极数据集路径 ---
TARGET_ROOT = r'E:\gravideo\final_dataset'

# 定义我们要检查的结构
splits = ['train', 'val', 'test_celeb']
categories = ['real', 'fake']

# 初始化统计字典
stats = {s: {c: 0 for c in categories} for s in splits}
total_images = 0

# --- 2. 扫描并统计 ---
if not os.path.exists(TARGET_ROOT):
    print(f"❌ 严重错误: 找不到数据集目录 {TARGET_ROOT}")
    exit()

for split in splits:
    for cat in categories:
        folder_path = os.path.join(TARGET_ROOT, split, cat)
        if os.path.exists(folder_path):
            # 只统计图片文件
            count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            stats[split][cat] = count
            total_images += count
        else:
            print(f"⚠️ 警告: 缺失关键文件夹 {folder_path}")

# --- 3. 生成精美的体检报告 ---
if total_images == 0:
    print("❌ 严重错误: 数据集里一张图片都没有！请检查前序脚本是否运行成功。")
    exit()

print(f"📁 数据集根目录: {TARGET_ROOT}")
print(f"🖼️ 总图片数量: {total_images} 张\n")

print("-" * 65)
print(f"{'划分集 (Split)':<15} | {'Real (真)':<10} | {'Fake (假)':<10} | {'小计 (Subtotal)':<15} | {'占比 (Ratio)':<10}")
print("-" * 65)

for split in splits:
    real_count = stats[split]['real']
    fake_count = stats[split]['fake']
    subtotal = real_count + fake_count
    ratio = (subtotal / total_images) * 100 if total_images > 0 else 0
    
    # 打印每一行的数据
    print(f"{split:<15} | {real_count:<10} | {fake_count:<10} | {subtotal:<15} | {ratio:.1f}%")

print("-" * 65)

# --- 4. 关键健康指标 (KPI) 诊断 ---
print("\n🩺 核心健康指标诊断结果:")

# 诊断 1：正负样本是否均衡
total_real = sum(stats[s]['real'] for s in splits)
total_fake = sum(stats[s]['fake'] for s in splits)
balance_ratio = total_real / total_fake if total_fake > 0 else float('inf')

if 0.95 <= balance_ratio <= 1.05:
    print(f"  ✅ 正负样本均衡度: 完美 (Real: {total_real} / Fake: {total_fake})")
else:
    print(f"  ❌ 正负样本失衡! (Real: {total_real} / Fake: {total_fake})，这会导致模型偏科！")

# 诊断 2：训练集是否有数据
if stats['train']['real'] > 0 and stats['train']['fake'] > 0:
    print(f"  ✅ 训练集状态: 正常加载 ({stats['train']['real'] + stats['train']['fake']} 张)")
else:
    print(f"  ❌ 训练集为空，模型将无粮可吃！")

# 诊断 3：跨域测试集是否存在
if stats['test_celeb']['real'] > 0 and stats['test_celeb']['fake'] > 0:
    print(f"  ✅ 跨域测试集: 就绪 (包含独立 Celeb-DF 数据)")
else:
    print("  ⚠️ 测试集异常，将无法评估跨域泛化能力！")

print("\n===============================================================")
print("💡 导师提示：如果上面全都是 ✅，你可以放心去启动训练脚本了！")
print("===============================================================")