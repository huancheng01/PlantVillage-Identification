import os
import cv2
import shutil
import random
from tqdm import tqdm  # 引入 tqdm 库

# 设置随机种子保证可复现性
random.seed(42)

# 数据集路径
img_dir = r'E:/studycode/py/12/Plant disease identification/archive/PlantVillage_for_object_detection/Dataset/images'
label_dir = r'E:/studycode/py/12/Plant disease identification/archive/PlantVillage_for_object_detection/Dataset/labels'
output_dir = r'E:/studycode/py/pythonProject/mypy_data/plantvillage_splitted'

# 记录错误日志的文件
error_log = "error_files.txt"

# 数据集划分比例
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# 数据集划分路径
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")

for split_dir in [train_dir, val_dir, test_dir]:
    os.makedirs(split_dir, exist_ok=True)

# 获取所有的标签文件
label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

# 打开日志文件
with open(error_log, "w") as log_file:
    for label_file in tqdm(label_files, desc="Processing images", unit="file"):
        img_file = os.path.join(img_dir, label_file.replace(".txt", ".jpg"))
        img = cv2.imread(img_file)

        if img is None:
            print(f"⚠️ 警告: 无法读取图像 {img_file}")
            log_file.write(f"无法读取图像: {img_file}\n")
            continue

        h, w, _ = img.shape
        label_path = os.path.join(label_dir, label_file)

        with open(label_path, "r") as f:
            for line_num, line in enumerate(f.readlines(), start=1):
                data = line.strip().split()

                if len(data) < 5:
                    print(f"🚨 错误: {label_file} 第 {line_num} 行格式错误 -> `{line.strip()}`")
                    log_file.write(f"格式错误: {label_file} 第 {line_num} 行 `{line.strip()}`\n")
                    continue

                try:
                    class_id = data[0]
                    x_center, y_center, width, height = map(float, data[1:])
                except ValueError as e:
                    print(f"🚨 解析错误: {label_file} 第 {line_num} 行 `{line.strip()}`，错误详情：{e}")
                    log_file.write(f"解析错误: {label_file} 第 {line_num} 行 `{line.strip()}`，错误详情：{e}\n")
                    continue

                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # 计算数据集划分
                rand_num = random.random()
                if rand_num < train_ratio:
                    split_dir = train_dir
                elif rand_num < train_ratio + val_ratio:
                    split_dir = val_dir
                else:
                    split_dir = test_dir

                class_dir = os.path.join(split_dir, class_id)
                os.makedirs(class_dir, exist_ok=True)
                save_path = os.path.join(class_dir, f"{label_file.replace('.txt', '')}_{x1}_{y1}.jpg")
                cv2.imwrite(save_path, crop)

print("✅ 数据集转换并划分完成！")
