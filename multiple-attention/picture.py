import pandas as pd

def process_reports():
    # 1. 读取原始CSV文件
    # 假设文件与脚本在同一目录下
    try:
        df_fake = pd.read_csv('batch_fake_report.csv')
        df_real = pd.read_csv('batch_real_report.csv')
    except FileNotFoundError:
        print("未找到文件，请确保 'batch_fake_report.csv' 和 'batch_real_report.csv' 在当前目录下。")
        return

    # 2. 定义所需提取的列与对应的目标表头名称
    columns_mapping = {
        '视频名称': '视频名称',
        '真实标签': '原视频的真假标签',
        'Base造假概率': '原模型给的造假概率',
        '集成后造假概率': '集成模型给的造假概率',
        '最终集成判定': '最终的判断结果',
        '耗时(s)': '耗时'
    }

    # 3. 提取指定的列并重命名表头
    df_fake_new = df_fake[list(columns_mapping.keys())].rename(columns=columns_mapping)
    df_real_new = df_real[list(columns_mapping.keys())].rename(columns=columns_mapping)

    # 4. 替换视频名称为匿名格式 (Sample_Fake_01, Sample_Real_05 等)
    df_fake_new['视频名称'] = [f"Sample_Fake_{i:02d}" for i in range(1, len(df_fake_new) + 1)]
    df_real_new['视频名称'] = [f"Sample_Real_{i:02d}" for i in range(1, len(df_real_new) + 1)]

    # 5. 生成并保存为新的CSV文件
    # 使用 utf-8-sig 编码可以防止在中文 Windows 系统下用 Excel 打开时出现乱码
    fake_output_name = 'formatted_fake_report.csv'
    real_output_name = 'formatted_real_report.csv'
    
    df_fake_new.to_csv(fake_output_name, index=False, encoding='utf-8-sig')
    df_real_new.to_csv(real_output_name, index=False, encoding='utf-8-sig')

    print(f"处理完成！\n已生成文件: {fake_output_name}\n已生成文件: {real_output_name}")

if __name__ == "__main__":
    process_reports()