import io
import torch
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request  # 添加 Request 导入
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="植物病害识别系统")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置模板（如果需要HTML页面）
templates = Jinja2Templates(directory="templates")

# 配置静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 加载模型
model = torch.load("best_resnet50.pth")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 保持原有的列表和映射不变
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

async def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

async def get_prediction(image_bytes):
    tensor = await transform_image(image_bytes)
    tensor = tensor.to(device)
    outputs = model(tensor)
    predicted_class = torch.argmax(outputs, dim=1).item()
    predicted_class = imagefolder_list[predicted_class]
    confidence = torch.max(torch.softmax(outputs, dim=1)).item()
    class_name = plant_list[predicted_class]
    return predicted_class, class_name, confidence

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):  # request 参数是必需的
    return templates.TemplateResponse(
        "index.html", 
        {"request": request}  # 传递给模板的上下文必须包含 request
    )

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        class_id, class_name, confidence = await get_prediction(img_bytes)
        return {
            "class_name": class_name,
            "confidence": confidence
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fast_api:app", host="0.0.0.0", port=5000, reload=True)