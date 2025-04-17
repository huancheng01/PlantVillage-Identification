# 使用的是 torchvision.datasets.ImageFolder，它会根据目录的字母顺序将类别映射到标签索引。
# 换句话说，train/4/ 目录下的图像并不一定会被分配给类别 4，具体分配给哪个类别，取决于文件夹的字母排序。
from torchvision.datasets import ImageFolder

# 加载训练集
train_dataset = ImageFolder(r"E:\studycode\py\pythonProject\mypy_data\plantvillage_splitted\train")
print("类别索引映射:", train_dataset.class_to_idx)
