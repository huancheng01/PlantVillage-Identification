import sys
import cv2
import requests
import numpy as np
import json
import io
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
                            QFrame, QSplitter, QSizePolicy, QGroupBox, QProgressBar, QScrollArea)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QPalette, QColor, QCursor
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QSize, QPropertyAnimation, QEasingCurve

# 设置Flask API的URL
API_URL = "http://127.0.0.1:5000/predict"

class PlantIdentificationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # 设置窗口标题和最小尺寸
        self.setWindowTitle("植物病害识别系统")
        self.setMinimumSize(1200, 800)
        # 设置窗口样式表，定义各种UI组件的样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            QPushButton:pressed {
                background-color: #3d8b40;
                transform: translateY(1px);
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
            QLabel {
                font-size: 14px;
                color: #333333;
            }
            QListWidget {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 8px;
                margin: 2px 0;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #e8f5e9;
                color: #2E7D32;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 16px;
                color: #2E7D32;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                margin-top: 16px;
                padding-top: 16px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
                background-color: white;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QProgressBar {
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                text-align: center;
                background-color: #f5f5f5;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 5px;
            }
        """)
        
        # 初始化变量 - 先于UI初始化
        self.current_image = None  # 当前显示的图像
        self.video_capture = None  # 视频捕获对象
        self.camera_active = False  # 摄像头状态
        self.timer = QTimer()  # 定时器，用于更新摄像头画面
        self.timer.timeout.connect(self.update_camera_frame)  # 连接定时器信号到更新函数
        
        # 病害类别列表
        self.plant_list = ["苹果黑星病","苹果黑腐病","苹果桧胶锈病","苹果健康叶","蓝莓健康叶","樱桃白粉病","樱桃健康叶",
                          "玉米灰斑病","玉米普通锈病","玉米叶枯病","玉米健康叶","葡萄黑腐病","葡萄黑痘病","葡萄叶枯病",
                          "葡萄健康叶","柑橘黄龙病","桃细菌性穿孔病","桃树健康叶","甜椒细菌性叶斑病","甜椒健康叶","土豆早疫病",
                          "土豆晚疫病","土豆健康叶","树莓健康叶","大豆健康叶","南瓜白粉病","草莓炭疽病","草莓健康叶",
                          "番茄细菌性斑疹病","番茄早疫病","番茄晚疫病","番茄叶霉病","番茄灰叶斑病","番茄二斑叶螨",
                          "番茄斑点病","番茄叶黄病毒病","番茄花叶病毒病","番茄健康叶"]
        
        # 类别映射
        self.imagefolder_list = {0: 0, 1: 1, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15, 8: 16, 9: 17,
                                10: 18, 11: 19, 12: 2, 13: 20, 14: 21, 15: 22, 16: 23, 17: 24, 18: 25,
                                19: 26, 20: 27, 21: 28, 22: 29, 23: 3, 24: 30, 25: 31, 26: 32, 27: 33,
                                28: 34, 29: 35, 30: 36, 31: 37, 32: 4, 33: 5, 34: 6, 35: 7, 36: 8, 37: 9}
        
        # 检查服务器连接状态
        self.server_available = self.check_server()
        
        # 初始化UI界面 - 放在最后
        self.init_ui()
        
    def check_server(self):
        """检查Flask服务器是否可用"""
        try:
            response = requests.get(API_URL.replace("/predict", ""), timeout=3)
            if response.status_code == 200:
                return True
            return False
        except:
            return False
        
    def init_ui(self):
        """初始化用户界面"""
        # 创建中央部件和主布局
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(20)
        
        # 添加标题和副标题
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        
        title_label = QLabel("植物病害识别系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 32px; 
            font-weight: bold; 
            color: #2E7D32; 
            margin-bottom: 15px;
            padding: 10px;
            background-color: white;
            border-radius: 10px;
            border-bottom: 3px solid #4CAF50;
        """)
        title_layout.addWidget(title_label)
        
        subtitle_label = QLabel("选择图片、视频或者摄像头进行识别")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("font-size: 16px; color: #666; margin-bottom: 20px;")
        title_layout.addWidget(subtitle_label)
        
        main_layout.addWidget(title_container)
        
        # 创建分割器，用于分割左右面板
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #e0e0e0;
            }
            QSplitter::handle:horizontal {
                width: 2px;
            }
        """)
        
        # 左侧面板 - 图像显示区域
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(15)
        
        # 图像预览区域
        image_group = QGroupBox("图像预览")
        image_group.setStyleSheet("""
            QGroupBox {
                padding: 20px;
            }
        """)
        image_layout = QVBoxLayout(image_group)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(500, 500)
        self.image_label.setStyleSheet("""
            background-color: #f0f0f0; 
            border-radius: 10px;
            border: 2px dashed #cccccc;
            padding: 10px;
        """)
        self.image_label.setText("未选择图像")
        
        image_layout.addWidget(self.image_label)
        left_layout.addWidget(image_group)
        
        # 图像源选择按钮组
        buttons_group = QGroupBox("图像源")
        buttons_layout = QHBoxLayout(buttons_group)
        buttons_layout.setContentsMargins(20, 20, 20, 20)
        buttons_layout.setSpacing(10)
        
        # 选择图片按钮
        self.btn_select_image = QPushButton("图片")
        self.btn_select_image.setIcon(QIcon.fromTheme("document-open"))
        self.btn_select_image.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_select_image.setMinimumHeight(50)
        self.btn_select_image.clicked.connect(self.select_image)
        
        # 选择视频按钮
        self.btn_select_video = QPushButton("视频")
        self.btn_select_video.setIcon(QIcon.fromTheme("video-x-generic"))
        self.btn_select_video.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_select_video.setMinimumHeight(50)
        self.btn_select_video.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.btn_select_video.clicked.connect(self.select_video)
        
        # 摄像头按钮
        self.btn_camera = QPushButton("摄像头")
        self.btn_camera.setIcon(QIcon.fromTheme("camera-photo"))
        self.btn_camera.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_camera.setMinimumHeight(50)
        self.btn_camera.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
            }
            QPushButton:hover {
                background-color: #e68a00;
            }
        """)
        self.btn_camera.clicked.connect(self.toggle_camera)
        
        buttons_layout.addWidget(self.btn_select_image)
        buttons_layout.addWidget(self.btn_select_video)
        buttons_layout.addWidget(self.btn_camera)
        
        left_layout.addWidget(buttons_group)
        
        # 识别按钮
        self.btn_identify = QPushButton("开始识别")
        self.btn_identify.setIcon(QIcon.fromTheme("system-search"))
        self.btn_identify.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_identify.clicked.connect(self.identify_plant)
        self.btn_identify.setEnabled(False)
        self.btn_identify.setMinimumHeight(60)
        self.btn_identify.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                padding: 15px 30px;
                font-size: 16px;
                border-radius: 8px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        
        left_layout.addWidget(self.btn_identify)
        
        # 右侧面板 - 结果显示区域
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(15)
        
        results_group = QGroupBox("识别结果")
        results_layout = QVBoxLayout(results_group)
        results_layout.setContentsMargins(20, 30, 20, 20)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.hide()
        results_layout.addWidget(self.progress_bar)
        
        # 结果列表（带滚动区域）
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        
        results_container = QWidget()
        results_container_layout = QVBoxLayout(results_container)
        
        self.results_list = QListWidget()
        self.results_list.setAlternatingRowColors(True)
        self.results_list.setStyleSheet("""
            QListWidget {
                font-size: 15px;
            }
            QListWidget::item {
                border-bottom: 1px solid #f0f0f0;
                padding: 10px;
            }
        """)
        
        results_container_layout.addWidget(self.results_list)
        scroll_area.setWidget(results_container)
        results_layout.addWidget(scroll_area)
        
        # 病害信息显示区域
        self.disease_info = QLabel("从列表中选择病害查看详细信息")
        self.disease_info.setWordWrap(True)
        self.disease_info.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.disease_info.setStyleSheet("""
            background-color: #f8f8f8;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            min-height: 100px;
        """)
        results_layout.addWidget(self.disease_info)
        
        right_layout.addWidget(results_group)
        
        # 状态显示区域
        status_container = QWidget()
        status_container.setStyleSheet("""
            background-color: white;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        """)
        status_layout = QHBoxLayout(status_container)
        
        status_icon = QLabel("📊")
        status_icon.setStyleSheet("font-size: 20px; padding-right: 10px;")
        status_layout.addWidget(status_icon)
        
        self.status_label = QLabel("准备就绪，可以开始识别")
        self.status_label.setStyleSheet("color: #555; font-style: italic; font-size: 14px;")
        status_layout.addWidget(self.status_label)
        
        right_layout.addWidget(status_container)
        
        # 将左右面板添加到分割器
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 400])
        
        main_layout.addWidget(splitter)
        
        self.setCentralWidget(central_widget)
        
        # 检查服务器状态并更新UI
        self.update_server_status()
    
    def update_server_status(self):
        """更新服务器状态显示"""
        if hasattr(self, 'status_label') and hasattr(self, 'btn_identify'):
            if self.check_server():
                self.server_available = True
                self.status_label.setText("准备就绪，服务器连接正常")
                self.btn_identify.setEnabled(self.current_image is not None)
            else:
                self.server_available = False
                self.status_label.setText("警告：无法连接到识别服务器！请确保Flask服务正在运行")
                self.status_label.setStyleSheet("color: #d32f2f; font-weight: bold; font-size: 14px;")
                self.btn_identify.setEnabled(False)
    
    def select_image(self):
        """选择图片文件并显示"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            # 如果摄像头正在运行，先停止
            if self.camera_active:
                self.toggle_camera()
                
            # 加载并显示图片
            self.current_image = cv2.imread(file_path)
            self.display_image(self.current_image)
            self.status_label.setText(f"已加载图片: {file_path.split('/')[-1]}")
            
            # 只有当服务器可用时才启用识别按钮
            self.btn_identify.setEnabled(self.server_available)
            
            # 如果服务器不可用，更新状态
            if not self.server_available:
                self.update_server_status()
            
            # 更新图片标签样式
            self.image_label.setStyleSheet("""
                background-color: #f0f0f0; 
                border-radius: 10px;
                border: 2px solid #4CAF50;
                padding: 10px;
            """)
    
    def select_video(self):
        """选择视频文件并播放"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.wmv)"
        )
        
        if file_path:
            # 如果摄像头正在运行，先停止
            if self.camera_active:
                self.toggle_camera()
                
            # 打开视频文件
            self.video_capture = cv2.VideoCapture(file_path)
            if self.video_capture.isOpened():
                self.timer.start(30)  # 每30ms更新一次
                self.camera_active = True
                self.btn_camera.setText("停止视频")
                self.status_label.setText(f"已加载视频: {file_path.split('/')[-1]}")
                self.btn_identify.setEnabled(True)
                
                # 更新按钮样式
                self.btn_select_video.setStyleSheet("""
                    QPushButton {
                        background-color: #f44336;
                    }
                    QPushButton:hover {
                        background-color: #d32f2f;
                    }
                """)
                self.btn_select_video.setText("停止视频")
            else:
                self.status_label.setText("错误：无法打开视频文件")
    
    def toggle_camera(self):
        """切换摄像头状态（开启/关闭）"""
        if self.camera_active:
            # 停止摄像头
            self.timer.stop()
            if self.video_capture:
                self.video_capture.release()
                self.video_capture = None
            self.camera_active = False
            self.btn_camera.setText("使用摄像头")
            self.status_label.setText("摄像头已停止")
            self.btn_identify.setEnabled(False)
            
            # 重置按钮样式
            self.btn_camera.setStyleSheet("""
                QPushButton {
                    background-color: #FF9800;
                }
                QPushButton:hover {
                    background-color: #e68a00;
                }
            """)
        else:
            # 启动摄像头
            self.video_capture = cv2.VideoCapture(0)
            if self.video_capture.isOpened():
                self.timer.start(30)  # 每30ms更新一次
                self.camera_active = True
                self.btn_camera.setText("停止摄像头")
                self.status_label.setText("摄像头已启动 - 可以开始识别")
                
                # 只有当服务器可用时才启用识别按钮
                self.btn_identify.setEnabled(self.server_available)
                
                # 如果服务器不可用，更新状态
                if not self.server_available:
                    self.update_server_status()
                
                # 更新按钮样式
                self.btn_camera.setStyleSheet("""
                    QPushButton {
                        background-color: #f44336;
                    }
                    QPushButton:hover {
                        background-color: #d32f2f;
                    }
                """)
            else:
                self.status_label.setText("错误：无法访问摄像头")
    
    def update_camera_frame(self):
        """更新摄像头画面"""
        if self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                self.current_image = frame
                self.display_image(frame)
            else:
                # 视频文件结束
                self.timer.stop()
                self.video_capture.release()
                self.video_capture = None
                self.camera_active = False
                self.btn_camera.setText("使用摄像头")
                self.status_label.setText("视频播放完成")
                self.btn_identify.setEnabled(False)
                
                # 重置按钮样式
                if self.sender() == self.btn_select_video:
                    self.btn_select_video.setStyleSheet("""
                        QPushButton {
                            background-color: #2196F3;
                        }
                        QPushButton:hover {
                            background-color: #0b7dda;
                        }
                    """)
                    self.btn_select_video.setText("选择视频")
    
    def display_image(self, cv_img):
        """显示图像"""
        if cv_img is None:
            return
            
        # 转换为RGB格式
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # 计算新尺寸，保持宽高比
        h, w, c = cv_img.shape
        label_width = self.image_label.width() - 20  # 考虑内边距
        label_height = self.image_label.height() - 20
        
        if w > h:
            new_width = min(label_width, w)
            new_height = int(h * (new_width / w))
        else:
            new_height = min(label_height, h)
            new_width = int(w * (new_height / h))
        
        # 转换为QImage，然后转换为QPixmap
        bytes_per_line = c * w
        q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # 缩放图像并设置到标签
        pixmap = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
    
    def identify_plant(self):
        """开始植物病害识别"""
        if self.current_image is None:
            self.status_label.setText("错误：没有图像可识别")
            return
            
        # 检查服务器是否可用
        if not self.server_available:
            self.update_server_status()
            if not self.server_available:
                self.status_label.setText("错误：无法连接到识别服务器！请确保Flask服务正在运行")
                return
        
        # 清除之前的结果
        self.results_list.clear()
        self.disease_info.setText("处理中...")
        
        # 显示进度条
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        
        # 更新状态
        self.status_label.setText("正在分析图像中的植物病害...")
        
        try:
            # 将OpenCV图像转换为PIL图像
            cv_img = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv_img)
            
            # 创建一个内存文件对象保存图像
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            # 更新进度条
            self.progress_bar.setValue(30)
            
            # 准备要发送的文件数据
            files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
            
            # 发送请求到Flask API
            self.status_label.setText("正在发送图像到服务器...")
            response = requests.post(API_URL, files=files)
            
            # 更新进度条
            self.progress_bar.setValue(70)
            
            if response.status_code == 200:
                # 解析返回的JSON结果
                result = response.json()
                disease_name = result.get('class_name')
                confidence = result.get('confidence')
                
                # 获取病害信息
                info = self.get_disease_info(disease_name)
                
                # 构建结果数据
                results = [{
                    "disease": disease_name,
                    "confidence": confidence,
                    "info": info
                }]
                
                # 显示结果
                self.show_results(results)
                self.status_label.setText("识别完成")
            else:
                self.status_label.setText(f"错误：服务器返回状态码 {response.status_code}")
                self.progress_bar.hide()
                
        except Exception as e:
            self.status_label.setText(f"错误：识别过程中发生错误 - {str(e)}")
            self.progress_bar.hide()
    
    def get_disease_info(self, disease_name):
        """获取病害的详细信息"""
        disease_info = {
            "苹果黑星病": "苹果黑星病是由真菌Venturia inaequalis引起的病害，主要危害叶片和果实。症状包括叶片出现黑褐色斑点，果实表面出现黑色病斑。防治方法包括使用杀菌剂、清除病叶和病果，以及保持果园通风透光。",
            "苹果黑腐病": "苹果黑腐病是由真菌Botryosphaeria obtusa引起的病害，会导致果实腐烂。症状包括果实表面出现黑色病斑，逐渐扩大并腐烂。防治方法包括及时清除病果、使用杀菌剂和保持果园清洁。",
            "苹果桧胶锈病": "苹果桧胶锈病是由真菌Gymnosporangium yamadae引起的病害，主要危害叶片和果实。症状包括叶片出现黄色斑点，果实表面出现锈斑。防治方法包括清除中间寄主桧柏、使用杀菌剂和保持果园通风。",
            "苹果健康叶": "叶片健康，无病害症状。建议定期检查，保持果园清洁，预防病害发生。",
            "蓝莓健康叶": "叶片健康，无病害症状。建议定期检查，保持果园清洁，预防病害发生。",
            "樱桃白粉病": "樱桃白粉病是由真菌Podosphaera clandestina引起的病害，主要危害叶片。症状包括叶片表面出现白色粉状物，严重时叶片卷曲。防治方法包括使用杀菌剂、清除病叶和保持果园通风。",
            "樱桃健康叶": "叶片健康，无病害症状。建议定期检查，保持果园清洁，预防病害发生。",
            "玉米灰斑病": "玉米灰斑病是由真菌Cercospora zeae-maydis引起的病害，主要危害叶片。症状包括叶片出现灰色斑点，严重时叶片枯死。防治方法包括使用杀菌剂、清除病叶和合理密植。",
            "玉米普通锈病": "玉米普通锈病是由真菌Puccinia sorghi引起的病害，主要危害叶片。症状包括叶片出现黄色或褐色锈斑。防治方法包括使用杀菌剂、清除病叶和选择抗病品种。",
            "玉米叶枯病": "玉米叶枯病是由真菌Exserohilum turcicum引起的病害，主要危害叶片。症状包括叶片出现褐色斑点，逐渐扩大导致叶片枯死。防治方法包括使用杀菌剂、清除病叶和合理密植。",
            "玉米健康叶": "叶片健康，无病害症状。建议定期检查，保持田间清洁，预防病害发生。",
            "葡萄黑腐病": "葡萄黑腐病是由真菌Guignardia bidwellii引起的病害，主要危害果实和叶片。症状包括果实表面出现黑色病斑，叶片出现褐色斑点。防治方法包括使用杀菌剂、清除病果和保持果园通风。",
            "葡萄黑痘病": "葡萄黑痘病是由真菌Elsinoe ampelina引起的病害，主要危害叶片和果实。症状包括叶片出现黑色小斑点，果实表面出现黑色病斑。防治方法包括使用杀菌剂、清除病叶和保持果园通风。",
            "葡萄叶枯病": "葡萄叶枯病是由真菌Phomopsis viticola引起的病害，主要危害叶片。症状包括叶片出现褐色斑点，逐渐扩大导致叶片枯死。防治方法包括使用杀菌剂、清除病叶和保持果园通风。",
            "葡萄健康叶": "叶片健康，无病害症状。建议定期检查，保持果园清洁，预防病害发生。",
            "柑橘黄龙病": "柑橘黄龙病是由细菌Candidatus Liberibacter asiaticus引起的病害，主要危害叶片和果实。症状包括叶片黄化、果实畸形。防治方法包括清除病树、防治木虱和使用抗生素。",
            "桃细菌性穿孔病": "桃细菌性穿孔病是由细菌Xanthomonas arboricola引起的病害，主要危害叶片。症状包括叶片出现水渍状斑点，逐渐扩大形成穿孔。防治方法包括使用杀菌剂、清除病叶和保持果园通风。",
            "桃树健康叶": "叶片健康，无病害症状。建议定期检查，保持果园清洁，预防病害发生。",
            "甜椒细菌性叶斑病": "甜椒细菌性叶斑病是由细菌Xanthomonas campestris引起的病害，主要危害叶片。症状包括叶片出现水渍状斑点，逐渐扩大形成坏死斑。防治方法包括使用杀菌剂、清除病叶和保持田间通风。",
            "甜椒健康叶": "叶片健康，无病害症状。建议定期检查，保持田间清洁，预防病害发生。",
            "土豆早疫病": "土豆早疫病是由真菌Alternaria solani引起的病害，主要危害叶片。症状包括叶片出现褐色斑点，逐渐扩大形成同心轮纹。防治方法包括使用杀菌剂、清除病叶和合理密植。",
            "土豆晚疫病": "土豆晚疫病是由真菌Phytophthora infestans引起的病害，主要危害叶片和块茎。症状包括叶片出现水渍状斑点，块茎出现褐色病斑。防治方法包括使用杀菌剂、清除病叶和合理密植。",
            "土豆健康叶": "叶片健康，无病害症状。建议定期检查，保持田间清洁，预防病害发生。",
            "树莓健康叶": "叶片健康，无病害症状。建议定期检查，保持果园清洁，预防病害发生。",
            "大豆健康叶": "叶片健康，无病害症状。建议定期检查，保持田间清洁，预防病害发生。",
            "南瓜白粉病": "南瓜白粉病是由真菌Podosphaera xanthii引起的病害，主要危害叶片。症状包括叶片表面出现白色粉状物，严重时叶片卷曲。防治方法包括使用杀菌剂、清除病叶和保持田间通风。",
            "草莓炭疽病": "草莓炭疽病是由真菌Colletotrichum acutatum引起的病害，主要危害果实和叶片。症状包括果实表面出现黑色病斑，叶片出现褐色斑点。防治方法包括使用杀菌剂、清除病果和保持田间通风。",
            "草莓健康叶": "叶片健康，无病害症状。建议定期检查，保持田间清洁，预防病害发生。",
            "番茄细菌性斑疹病": "番茄细菌性斑疹病是由细菌Pseudomonas syringae引起的病害，主要危害叶片。症状包括叶片出现水渍状斑点，逐渐扩大形成坏死斑。防治方法包括使用杀菌剂、清除病叶和保持田间通风。",
            "番茄早疫病": "番茄早疫病是由真菌Alternaria solani引起的病害，主要危害叶片。症状包括叶片出现褐色斑点，逐渐扩大形成同心轮纹。防治方法包括使用杀菌剂、清除病叶和合理密植。",
            "番茄晚疫病": "番茄晚疫病是由真菌Phytophthora infestans引起的病害，主要危害叶片和果实。症状包括叶片出现水渍状斑点，果实表面出现褐色病斑。防治方法包括使用杀菌剂、清除病叶和合理密植。",
            "番茄叶霉病": "番茄叶霉病是由真菌Fulvia fulva引起的病害，主要危害叶片。症状包括叶片出现黄色斑点，背面出现灰色霉层。防治方法包括使用杀菌剂、清除病叶和保持田间通风。",
            "番茄灰叶斑病": "番茄灰叶斑病是由真菌Stemphylium solani引起的病害，主要危害叶片。症状包括叶片出现灰色斑点，逐渐扩大导致叶片枯死。防治方法包括使用杀菌剂、清除病叶和保持田间通风。",
            "番茄二斑叶螨": "番茄二斑叶螨是由螨类Tetranychus urticae引起的虫害，主要危害叶片。症状包括叶片出现黄色斑点，严重时叶片枯死。防治方法包括使用杀螨剂、清除病叶和保持田间通风。",
            "番茄斑点病": "番茄斑点病是由真菌Septoria lycopersici引起的病害，主要危害叶片。症状包括叶片出现褐色斑点，逐渐扩大形成坏死斑。防治方法包括使用杀菌剂、清除病叶和保持田间通风。",
            "番茄叶黄病毒病": "番茄叶黄病毒病是由病毒引起的病害，主要危害叶片。症状包括叶片黄化、卷曲和畸形。防治方法包括防治传毒媒介、清除病株和使用抗病品种。",
            "番茄花叶病毒病": "番茄花叶病毒病是由病毒引起的病害，主要危害叶片。症状包括叶片出现花叶症状，严重时植株矮化。防治方法包括防治传毒媒介、清除病株和使用抗病品种。",
            "番茄健康叶": "叶片健康，无病害症状。建议定期检查，保持田间清洁，预防病害发生。"
        }
        return disease_info.get(disease_name, "暂无详细信息")
    
    def show_results(self, results):
        """显示识别结果"""
        # 完成进度条
        self.progress_bar.setValue(100)
        QTimer.singleShot(500, lambda: self.progress_bar.hide())
        
        for result in results:
            disease = result["disease"]
            confidence = result["confidence"]
            
            item = QListWidgetItem(f"{disease}: {confidence*100:.1f}%")
            item.setData(Qt.UserRole, result["info"])
            
            # 根据置信度设置背景颜色
            if confidence > 0.7:
                item.setBackground(QColor(200, 255, 200))  # 浅绿色
                item.setForeground(QColor(0, 100, 0))      # 深绿色文字
            elif confidence > 0.4:
                item.setBackground(QColor(255, 255, 200))  # 浅黄色
                item.setForeground(QColor(128, 128, 0))    # 橄榄色文字
            else:
                item.setBackground(QColor(255, 230, 230))  # 浅红色
                item.setForeground(QColor(128, 0, 0))      # 深红色文字
                
            self.results_list.addItem(item)
        
        # 连接项目选择到显示病害信息
        self.results_list.itemClicked.connect(self.show_disease_info)
        
        # 默认选择第一个项目
        if self.results_list.count() > 0:
            self.results_list.setCurrentRow(0)
            self.show_disease_info(self.results_list.item(0))
        
        self.status_label.setText("识别完成 - 点击病害查看详细信息")
    
    def show_disease_info(self, item):
        """显示病害详细信息"""
        info = item.data(Qt.UserRole)
        disease_name = item.text().split(':')[0]
        
        self.disease_info.setText(f"<h3>{disease_name}</h3><p>{info}</p>")

if __name__ == "__main__":
    # 创建应用程序实例
    app = QApplication(sys.argv)
    window = PlantIdentificationApp()
    window.show()
    sys.exit(app.exec_())
