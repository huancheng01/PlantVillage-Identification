import torch
from torchvision import transforms
from PIL import Image

# 加载模型
model = torch.load("best_resnet50.pth")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

imagefolder_list = {0: 0, 1: 1, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15, 8: 16, 9: 17,
                    10: 18, 11: 19, 12: 2, 13: 20, 14: 21, 15: 22, 16: 23, 17: 24, 18: 25,
                    19: 26, 20: 27, 21: 28, 22: 29, 23: 3, 24: 30, 25: 31, 26: 32, 27: 33,
                    28: 34, 29: 35, 30: 36, 31: 37, 32: 4, 33: 5, 34: 6, 35: 7, 36: 8, 37: 9}


plant_list = ["苹果黑星病","苹果黑腐病","苹果桧胶锈病","苹果健康叶","蓝莓健康叶","樱桃白粉病","樱桃健康叶",
              "玉米灰斑病","玉米普通锈病","玉米叶枯病","玉米健康叶","葡萄黑腐病","葡萄黑痘病","葡萄叶枯病",
              "葡萄健康叶","柑橘黄龙病","桃细菌性穿孔病","桃树健康叶","甜椒细菌性叶斑病","甜椒健康叶","土豆早疫病",
              "土豆晚疫病","土豆健康叶","树莓健康叶","大豆健康叶","南瓜白粉病","草莓炭疽病","草莓健康叶",
              "番茄细菌性斑疹病","番茄早疫病","番茄晚疫病","番茄叶霉病","番茄灰叶斑病","番茄二斑叶螨",
              "番茄斑点病","番茄叶黄病毒病","番茄花叶病毒病","番茄健康叶"]

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 加载图像
image_path = r"E:\studycode\py\pythonProject\mypy_data\plantvillage_splitted\train\4\BLHE_image (1)_22_23.jpg"
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# 推理
with torch.no_grad():
    output = model(image)
    probs = torch.softmax(output, dim=1)
    confidence, predicted_class = torch.max(probs, 1)

classs_id = imagefolder_list[predicted_class.item()]
print(f"预测类别: {classs_id}, 置信度: {confidence.item():.2f}")
print(f"名称：{plant_list[classs_id]}")

