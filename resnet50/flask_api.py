import io
import json
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS  # 导入 CORS

app = Flask(__name__)
CORS(app)  # 允许所有来源的跨域请求，默认允许所有

# 加载模型
model = torch.load("best_resnet50.pth")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

plant_list = ["苹果黑星病","苹果黑腐病","苹果桧胶锈病","苹果健康叶","蓝莓健康叶","樱桃白粉病","樱桃健康叶",
              "玉米灰斑病","玉米普通锈病","玉米叶枯病","玉米健康叶","葡萄黑腐病","葡萄黑痘病","葡萄叶枯病",
              "葡萄健康叶","柑橘黄龙病","桃细菌性穿孔病","桃树健康叶","甜椒细菌性叶斑病","甜椒健康叶","土豆早疫病",
              "土豆晚疫病","土豆健康叶","树莓健康叶","大豆健康叶","南瓜白粉病","草莓炭疽病","草莓健康叶",
              "番茄细菌性斑疹病","番茄早疫病","番茄晚疫病","番茄叶霉病","番茄灰叶斑病","番茄二斑叶螨",
              "番茄斑点病","番茄叶黄病毒病","番茄花叶病毒病","番茄健康叶"]

imagefolder_list = {0: 0, 1: 1, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15, 8: 16, 9: 17,
                    10: 18, 11: 19, 12: 2, 13: 20, 14: 21, 15: 22, 16: 23, 17: 24, 18: 25,
                    19: 26, 20: 27, 21: 28, 22: 29, 23: 3, 24: 30, 25: 31, 26: 32, 27: 33,
                    28: 34, 29: 35, 30: 36, 31: 37, 32: 4, 33: 5, 34: 6, 35: 7, 36: 8, 37: 9}

def transform_image(image_bytes):
    # 定义图像预处理方法
    my_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    tensor = tensor.to(device)
    outputs = model(tensor)
    # 获取预测的类别及其置信度
    predicted_class = torch.argmax(outputs, dim=1).item()
    predicted_class = imagefolder_list[predicted_class]     #将类别映射为正值
    confidence = torch.max(torch.softmax(outputs, dim=1)).item()
    # 使用类别索引查找类别名称
    class_name = plant_list[predicted_class]  # 从 plant_list 获取类别名称
    return predicted_class, class_name, confidence

# 添加根路由处理

@app.route('/')
def home():
    return render_template('index.html')  # 让 Flask 提供 HTML 页面

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({"message": "请使用 POST 方法上传图片进行预测"}), 200
    if request.method == 'POST':
        # 获取上传的文件
        file = request.files['file']
        img_bytes = file.read()
        # 获取预测结果
        class_id, class_name, confidence = get_prediction(image_bytes=img_bytes)
        return jsonify({
            'class_name': class_name,
            'confidence': confidence
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
