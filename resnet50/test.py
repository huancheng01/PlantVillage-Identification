import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, models

# 直接加载整个模型
model = torch.load("best_resnet50.pth")
model.eval()  # 切换到评估模式

device = torch.device("cuda") 
model.to(device)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = torchvision.datasets.ImageFolder(
    root="E:/wjx/py/mydata/plantvillage_splitted/test",
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

correct = 0
total = 0
criterion = torch.nn.CrossEntropyLoss()
running_loss = 0.0

with torch.no_grad():  # 评估时不需要计算梯度
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)  # 确保数据和模型在同一设备上

        outputs = model(images)  # 获取模型输出
        _, predicted = torch.max(outputs, 1)  # 获取最大概率的类别

        total += labels.size(0)
        correct += (predicted == labels).sum().item()  # 统计正确的预测数量

        # 计算损失
        loss = criterion(outputs, labels)
        running_loss += loss.item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
print(f'Test Loss: {running_loss / len(test_loader):.4f}')
