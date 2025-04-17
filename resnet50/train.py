import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

# 设置 TORCH_HOME
os.environ['TORCH_HOME'] = 'E:/wjx/py/resnet50'  # 自定义缓存路径

# 超参数
batch_size = 32
lr = 0.001
num_epochs = 50
patience = 5  # Early Stopping的耐心值

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = torchvision.datasets.ImageFolder(root="E:/wjx/py/mydata/plantvillage_splitted/train",transform=transform)
val_dataset = torchvision.datasets.ImageFolder(root="E:/wjx/py/mydata/plantvillage_splitted/val",transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root="E:/wjx/py/mydata/plantvillage_splitted/test",transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 加载ResNet50
device = torch.device("cuda")
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # 增加 L2 正则化

# 学习率调度器（ReduceLROnPlateau）
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


# Early Stopping 类
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print("Early stopping triggered!")
            return True
        return False


# 初始化 Early Stopping
early_stopping = EarlyStopping(patience=5)

# 创建 TensorBoard 的 writer
writer = SummaryWriter(log_dir='./logs')

# 训练循环
best_loss = np.inf  # 记录最佳验证损失
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 计算准确率
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")
    # 获取当前学习率
    print(f"Current Learning Rate: {scheduler.optimizer.param_groups[0]['lr']}")
    # 验证阶段
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    writer.add_scalar('Loss/val', val_loss / len(val_loader), epoch)
    writer.add_scalar('Accuracy/val', val_accuracy, epoch)

    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # ReduceLROnPlateau 调整学习率
    scheduler.step(val_loss)

    # 保存最佳模型
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model, "best_resnet50.pth")
        print("Best model saved!")

    # 检查 Early Stopping 条件
    if early_stopping.step(val_loss / len(val_loader)):
        break  # 终止训练


# 关闭 TensorBoard writer
writer.close()
