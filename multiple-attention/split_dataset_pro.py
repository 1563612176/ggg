#划分最终微调模型的数据集的图片
import os
import random
import shutil
from tqdm import tqdm

print("============== 📦 终极数据集混合器 (严格 80:20 版) 启动 ==============")

# --- 1. 路径配置 ---
FF_CROP_ROOT = r'E:\gravideo\cropped_faces' 
CELEB_ROOT = r'E:\gravideo\celebdf-v2image-dataset\Celeb_V2'
TARGET_ROOT = r'E:\gravideo\final_dataset'

def get_all_images(directory_list):
    img_paths = []
    for directory in directory_list:
        if not os.path.exists(directory):
            continue
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_paths.append(os.path.join(root, file))
    return img_paths

def move_samples(source_paths, target_dir, num_samples, desc):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    sample_size = min(num_samples, len(source_paths))
    if sample_size < num_samples:
        print(f"⚠️ 警告: {desc} 兵力不足！需要 {num_samples}，只有 {sample_size}")
        
    selected = random.sample(source_paths, sample_size)
    for path in tqdm(selected, desc=desc, leave=False):
        parent_name = os.path.basename(os.path.dirname(path))
        file_name = os.path.basename(path)
        new_name = f"{parent_name}_{file_name}"
        shutil.copy(path, os.path.join(target_dir, new_name))
    
    return [p for p in source_paths if p not in selected]

# --- 2. 收集双方兵力 ---
print("\n正在扫描兵力...")
ff_real_dirs = [os.path.join(FF_CROP_ROOT, 'original')]
ff_fake_dirs = [
    os.path.join(FF_CROP_ROOT, 'Deepfakes'),
    os.path.join(FF_CROP_ROOT, 'Face2Face'),
    os.path.join(FF_CROP_ROOT, 'FaceSwap'),
    os.path.join(FF_CROP_ROOT, 'NeuralTextures'),
    os.path.join(FF_CROP_ROOT, 'FaceShifter')
]

ff_real_images = get_all_images(ff_real_dirs)
ff_fake_images = get_all_images(ff_fake_dirs)
celeb_train_real = get_all_images([os.path.join(CELEB_ROOT, 'Train', 'real')])
celeb_train_fake = get_all_images([os.path.join(CELEB_ROOT, 'Train', 'fake')])
celeb_test_real = get_all_images([os.path.join(CELEB_ROOT, 'Test', 'real')])
celeb_test_fake = get_all_images([os.path.join(CELEB_ROOT, 'Test', 'fake')])
print(f"✅ 兵力集结完毕！FF++ (真:{len(ff_real_images)} 假:{len(ff_fake_images)})")

# --- 3. 开始按【严格 80:20 黄金比例】混合搬运 ---
print("\n🚀 开始构建终极训练场...")

# 🔴【Train 训练集】: FF++ 2000张(80%) + Celeb-DF 500张(20%)
ff_real_images = move_samples(ff_real_images, os.path.join(TARGET_ROOT, 'train', 'real'), 2000, "Train-Real (FF++ 80%)")
ff_fake_images = move_samples(ff_fake_images, os.path.join(TARGET_ROOT, 'train', 'fake'), 2000, "Train-Fake (FF++ 80%)")
_ = move_samples(celeb_train_real, os.path.join(TARGET_ROOT, 'train', 'real'), 500, "Train-Real (Celeb 20%)")
_ = move_samples(celeb_train_fake, os.path.join(TARGET_ROOT, 'train', 'fake'), 500, "Train-Fake (Celeb 20%)")

# 🟡【Val 验证集】: 消耗剩下的 FF++ (各 500张)
ff_real_images = move_samples(ff_real_images, os.path.join(TARGET_ROOT, 'val', 'real'), 500, "Val-Real (FF++)")
ff_fake_images = move_samples(ff_fake_images, os.path.join(TARGET_ROOT, 'val', 'fake'), 500, "Val-Fake (FF++)")

# 🟢【Test 测试集】: 抽 500张 Celeb-DF
_ = move_samples(celeb_test_real, os.path.join(TARGET_ROOT, 'test_celeb', 'real'), 500, "Test-Real (Celeb)")
_ = move_samples(celeb_test_fake, os.path.join(TARGET_ROOT, 'test_celeb', 'fake'), 500, "Test-Fake (Celeb)")

print("\n🎉 严格 80:20 的完美数据集组装完毕！位置: E:\gravideo\final_dataset")