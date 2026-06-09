import sys
import os
from ultralytics import YOLO
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QGridLayout, QSpinBox, QListWidget, 
    QMessageBox, QFileDialog, QTextEdit, QScrollArea, QRadioButton,
    QTabWidget, QSlider
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QPoint
from PyQt5.QtGui import QFont, QPainter, QColor, QLinearGradient, QBrush, QImage, QPixmap, QPen

# 传统蓝牙库 (HC-05/HC-06)
try:
    import bluetooth
    BLUETOOTH_AVAILABLE = True
except ImportError:
    BLUETOOTH_AVAILABLE = False
    print("警告: pybluez未安装，蓝牙功能不可用")


# ========== seal.py 视觉识别逻辑（完全照搬） ==========
def perspective_transform(image, src_points, dst_points=None, output_size=None):
    """透视变换"""
    src = np.array(src_points, dtype=np.float32)
    if src.shape != (4, 2):
        raise ValueError("src_points 必须是形状为 (4, 2) 的数组")

    if dst_points is None:
        (tl, tr, br, bl) = src
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        max_width = max(int(width_top), int(width_bottom))

        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        max_height = max(int(height_left), int(height_right))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)
    else:
        dst = np.array(dst_points, dtype=np.float32)
        if dst.shape != (4, 2):
            raise ValueError("dst_points 必须是形状为 (4, 2) 的数组")

    M = cv2.getPerspectiveTransform(src, dst)

    if output_size is None:
        width = int(np.max(dst[:, 0]) - np.min(dst[:, 0]) + 1)
        height = int(np.max(dst[:, 1]) - np.min(dst[:, 1]) + 1)
        output_size = (width, height)

    warped = cv2.warpPerspective(image, M, output_size)
    return warped

def dect_dir(crop_img):
    """质心检测箭头方向（完全照搬seal.py）"""
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown"

    cnt = max(contours, key=cv2.contourArea)

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return "unknown"
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    max_dist = 0
    tip = (cx, cy)
    for point in cnt:
        pt = point[0]
        dist = (pt[0] - cx) ** 2 + (pt[1] - cy) ** 2
        if dist > max_dist:
            max_dist = dist
            tip = (pt[0], pt[1])

    dx = tip[0] - cx
    dy = tip[1] - cy
    if abs(dx) > abs(dy):
        direction = "right" if dx > 0 else "left"
    else:
        direction = "down" if dy > 0 else "up"
    return direction

def get_lines(img):
    """检测网格线并切分格子（完全照搬seal.py）"""
    h, w = img.shape[:2]
    white = np.ones((h, w, 3), dtype=np.uint8) * 255
    white_copy = white.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 15)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    lines = cv2.HoughLinesP(closed, rho=1, theta=np.pi/180, threshold=120, minLineLength=800, maxLineGap=80)
    i = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(white, (x1, y1), (x2, y2), (0, 0, 0), thickness=5, lineType=cv2.LINE_AA)
        i = i + 1
    print(i)

    h_lines, v_lines = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 20 or abs(angle) > 160:
                h_lines.append((x1, y1, x2, y2))
            elif 70 < abs(angle) < 110:
                v_lines.append((x1, y1, x2, y2))

    def merge_lines(lines, axis='y'):
        if not lines:
            return []

        centers = []
        for x1, y1, x2, y2 in lines:
            cx, cy = (x1+x2)/2, (y1+y2)/2
            centers.append((cx, cy))
        centers = np.array(centers)

        idx = 1 if axis == 'y' else 0
        sorted_centers = centers[centers[:, idx].argsort()]

        merged = []
        cluster = [sorted_centers[0]]
        for i in range(1, len(sorted_centers)):
            if abs(sorted_centers[i][idx] - sorted_centers[i-1][idx]) < 20:
                cluster.append(sorted_centers[i])
            else:
                merged.append(np.mean(cluster, axis=0))
                cluster = [sorted_centers[i]]
        merged.append(np.mean(cluster, axis=0))
        return merged

    h_merged = merge_lines(h_lines, axis='y')
    v_merged = merge_lines(v_lines, axis='x')
    h_merged = sorted(h_merged, key=lambda p: p[1])
    v_merged = sorted(v_merged, key=lambda p: p[0])
    print(v_merged)
    print(h_merged)
    
    for v in v_merged:
        x_v = int(v[0])
        for h in h_merged:
            y_h = int(h[1])
            cv2.circle(white_copy, (x_v, y_h), 3, (0, 0, 0), -1)
    
    grid_points = []
    for v in v_merged:
        x_v = int(v[0])
        for h in h_merged:
            y_h = int(h[1])
            grid_points.append((x_v, y_h))

    grid_points.sort(key=lambda p: (p[1], p[0]))

    rows = len(h_merged)
    cols = len(v_merged)
    if len(grid_points) == rows * cols:
        point_matrix = np.array(grid_points).reshape(rows, cols, 2)
    else:
        print("交点数量与行列数不匹配")
        return []

    cells = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            tl = point_matrix[i][j]
            tr = point_matrix[i][j+1]
            bl = point_matrix[i+1][j]
            br = point_matrix[i+1][j+1]

            x_min, x_max = int(min(tl[0], tr[0], bl[0], br[0])), int(max(tl[0], tr[0], bl[0], br[0]))
            y_min, y_max = int(min(tl[1], tr[1], bl[1], br[1])), int(max(tl[1], tr[1], bl[1], br[1]))
            cell_img = img[y_min:y_max, x_min:x_max]
            cells.append(cell_img)

    print(f"切分出 {len(cells)} 个格子")
    return cells

def model_dect(model, roi, use_centroid_for_arrows=False):
    """模型检测（完全照搬seal.py）
    
    模型返回标签格式：
    - 数字：1, 2, 3, 4, 5, 6, 7, 8 (阿拉伯数字字符串)
    - 方向：up, down, left, right (小写字符串)
    
    映射规则：
    - up -> FORWARD
    - down -> BACKWARD
    - left -> LEFT (质心模式下可能返回up/down)
    - right -> RIGHT (质心模式下可能返回up/down)
    - 数字直接返回int类型
    """
    result = model.predict(source=roi)
    if result is not None:
        for r in result:
            probs = r.probs
            top1_idx = probs.top1
            top1_conf = probs.top1conf
            label = r.names[top1_idx]
            print(f"预测类别: {label}，置信度: {top1_conf:.4f}")
            
            # 数字标签处理 (1-8的阿拉伯数字字符串)
            if label in ['1', '2', '3', '4', '5', '6', '7', '8']:
                return None, int(label)
            
            # 方向标签处理
            elif label == 'up':
                return 'FORWARD', None
            elif label == 'down':
                return 'BACKWARD', None
            elif label == 'left':
                if use_centroid_for_arrows:
                    # 使用质心检测
                    direction = dect_dir(roi)
                    print(f"质心检测方向: {direction}")
                    # 质心检测返回: right, left, up, down
                    dir_map = {'right': 'RIGHT', 'left': 'LEFT', 'up': 'FORWARD', 'down': 'BACKWARD'}
                    return dir_map.get(direction, direction.upper()), None
                else:
                    return 'LEFT', None
            elif label == 'right':
                if use_centroid_for_arrows:
                    # 使用质心检测
                    direction = dect_dir(roi)
                    print(f"质心检测方向: {direction}")
                    # 质心检测返回: right, left, up, down
                    dir_map = {'right': 'RIGHT', 'left': 'LEFT', 'up': 'FORWARD', 'down': 'BACKWARD'}
                    return dir_map.get(direction, direction.upper()), None
                else:
                    return 'RIGHT', None
        
        return None, None
    return None, None

def divide_dect(img, model, use_centroid_for_arrows=False):
    """分割检测（完全照搬seal.py）"""
    hh, ww, _ = img.shape
    dis = 10
    img = img[dis:hh-dis, dis:ww-dis]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_h, img_w = gray.shape
    count = 0
    results = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        if 300 < area < 5000 and img_w*0.1 < w < img_w * 0.7 and img_h*0.1 < h < img_h * 0.7:
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img_w, x + w + pad)
            y2 = min(img_h, y + h + pad)

            cropped_img = img[y1:y2, x1:x2]
            direction, number = model_dect(model, cropped_img, use_centroid_for_arrows)
            results.append({"direction": direction, "number": number})
            count += 1

    if count == 0:
        print("未提取到有效对象")
    
    return results if results else None
# ========== 传统蓝牙连接管理 (HC-05/HC-06) ==========
class BluetoothManager:
    def __init__(self):
        self.bluetooth_socket = None
        
    def discover_devices(self):
        """搜索蓝牙设备"""
        devices = []
        try:
            discovered = bluetooth.discover_devices(duration=8, lookup_names=True, flush_cache=True)
            for addr, name in discovered:
                devices.append((name, addr))
        except Exception as e:
            print(f"蓝牙扫描失败: {str(e)}")
        return devices
        
    def connect_device(self, address):
        """连接蓝牙设备"""
        try:
            self.bluetooth_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self.bluetooth_socket.connect((address, 1))
            return True, ""
        except Exception as e:
            self.bluetooth_socket = None
            return False, str(e)
            
    def disconnect_device(self):
        """断开蓝牙连接"""
        if self.bluetooth_socket:
            try:
                self.bluetooth_socket.close()
            except:
                pass
            self.bluetooth_socket = None
            
    def is_connected(self):
        """检查是否已连接"""
        return self.bluetooth_socket is not None
        
    def send_data(self, data):
        """发送数据"""
        if not self.bluetooth_socket:
            return False, "蓝牙未连接"
            
        try:
            cmd_bytes = (data + '\n').encode('utf-8')
            self.bluetooth_socket.send(cmd_bytes)
            return True, ""
        except Exception as e:
            return False, str(e)


class MazeGridWidget(QWidget):
    """7×7迷宫显示控件"""
    cell_clicked = pyqtSignal(int, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.maze_data = None
        self.start_point = None
        self.path_cells = []
        self.cell_size = 60
        self.setFixedSize(420, 420)
        
    def set_maze_data(self, data):
        self.maze_data = data
        self.path_cells = []
        self.update()
        
    def set_start_point(self, row, col):
        self.start_point = (row, col)
        self.update()
        
    def set_path_cells(self, cells):
        self.path_cells = cells
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        for row in range(7):
            for col in range(7):
                x = col * self.cell_size
                y = row * self.cell_size
                
                rect_color = QColor(245, 250, 255)
                if self.start_point == (row, col):
                    rect_color = QColor(100, 200, 100, 150)
                elif (row, col) in self.path_cells:
                    rect_color = QColor(255, 200, 100, 150)
                    
                painter.fillRect(x, y, self.cell_size, self.cell_size, QBrush(rect_color))
                painter.drawRect(x, y, self.cell_size, self.cell_size)
                
                if self.maze_data and len(self.maze_data) > row and len(self.maze_data[row]) > col:
                    cell = self.maze_data[row][col]
                    number = cell.get('number')
                    direction = cell.get('direction')
                    
                    if number is not None:
                        painter.setPen(QColor(50, 50, 50))
                        painter.setFont(QFont('Arial', 20, QFont.Bold))
                        painter.drawText(x + 10, y + self.cell_size//2 + 5, str(number))
                    
                    if direction and direction != '':
                        painter.setPen(QColor(50, 50, 50))
                        painter.setFont(QFont('Arial', 14))
                        arrow_text = {'FORWARD': '↑', 'BACKWARD': '↓', 'LEFT': '←', 'RIGHT': '→'}
                        if direction in arrow_text:
                            painter.drawText(x + self.cell_size - 25, y + self.cell_size//2 + 5, arrow_text[direction])
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x = event.x()
            y = event.y()
            col = x // self.cell_size
            row = y // self.cell_size
            if 0 <= row < 7 and 0 <= col < 7:
                self.cell_clicked.emit(row, col)

class SelectPointLabel(QLabel):
    """支持鼠标点击选点、绘制标记点的图片标签"""
    # ========== 关键修复1：信号定义在【类顶层】（类属性），不是__init__里 ==========
    # 对应参数：点序号, 原图X, 原图Y  一共3个int参数
    clicked_point = pyqtSignal(int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.origin_img = None       # 原始OpenCV图像
        self.scale_points = []      # 预览图上的缩放坐标(用于绘图)
        self.origin_points = []     # 对应原图真实坐标
        self.point_order = 0        # 选点顺序：0左上 1右上 2右下 3左下
        self.point_color = [
            QColor(255, 0, 0),    # 红-左上
            QColor(0, 128, 0),    # 绿-右上
            QColor(0, 0, 255),    # 蓝-右下
            QColor(255, 165, 0)   # 橙-左下
        ]
        self.point_radius = 6

    def set_origin_image(self, cv_img):
        """设置原始OpenCV图像，并清空选点"""
        self.origin_img = cv_img
        self.scale_points.clear()
        self.origin_points.clear()
        self.point_order = 0
        self.update()

    def clear_points(self):
        """清空所有选点"""
        self.scale_points.clear()
        self.origin_points.clear()
        self.point_order = 0
        self.update()

    def mousePressEvent(self, event):
        """鼠标点击选点"""
        if event.button() != Qt.LeftButton or self.origin_img is None:
            return

        # 1. 获取点击位置(Label内缩放坐标)
        click_x = event.x()
        click_y = event.y()
        self.scale_points.append((click_x, click_y))

        # 2. 缩放坐标 → 原始图像真实坐标（核心换算）
        label_w = self.width()
        label_h = self.height()
        img_h, img_w = self.origin_img.shape[:2]

        # 保持等比例缩放，计算实际显示区域偏移
        scale = min(label_w / img_w, label_h / img_h)
        disp_w = img_w * scale
        disp_h = img_h * scale
        offset_x = (label_w - disp_w) / 2
        offset_y = (label_h - disp_h) / 2

        # 换算到原图像素坐标
        real_x = int((click_x - offset_x) / scale)
        real_y = int((click_y - offset_y) / scale)
        real_x = max(0, min(img_w - 1, real_x))
        real_y = max(0, min(img_h - 1, real_y))

        self.origin_points.append((real_x, real_y))

        # 3. 发射信号：3个参数(序号, x, y)，和信号定义严格对应
        self.clicked_point.emit(self.point_order, real_x, real_y)

        # 4. 切换下一个选点，4个点一轮回
        self.point_order = (self.point_order + 1) % 4
        self.update()

    def paintEvent(self, event):
        """绘制标记点（已移除QPoint，兼容旧导入）"""
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 遍历所有已选点绘制圆圈
        for idx, (x, y) in enumerate(self.scale_points):
            color = self.point_color[idx]
            painter.setPen(QPen(color, 2))
            painter.setBrush(color)
            # 直接传坐标，不使用QPoint，规避导入问题
            painter.drawEllipse(x, y, self.point_radius, self.point_radius)
            # 绘制点序号
            painter.setPen(Qt.white)
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.drawText(x-3, y+3, str(idx+1))

class FinalApp(QWidget):
    def __init__(self):
        super().__init__()
        self.ble_devices = []
        self.bluetooth_manager = BluetoothManager()
        self.cap = None
        self.detected_image = None
        self.yolo_model = None
        self.maze_data = None
        self.start_point = None
        self.path = []
        self.commands = []
        self.return_commands = []
        
        self.initUI()
        
        # 初始化蓝牙管理器
        # 蓝牙扫描在refresh_bluetooth_devices中执行
        
    def on_image_point_selected(self, point_idx, x, y):
        """
        图片选点回调
        :param point_idx: 点序号 0=左上 1=右上 2=右下 3=左下
        :param x: 原图X坐标
        :param y: 原图Y坐标
        """
        if point_idx == 0:
            # 左上角
            self.spin_tl_x.setValue(x)
            self.spin_tl_y.setValue(y)
        elif point_idx == 1:
            # 右上角
            self.spin_tr_x.setValue(x)
            self.spin_tr_y.setValue(y)
        elif point_idx == 2:
            # 右下角
            self.spin_br_x.setValue(x)
            self.spin_br_y.setValue(y)
        elif point_idx == 3:
            # 左下角
            self.spin_bl_x.setValue(x)
            self.spin_bl_y.setValue(y)

    def initUI(self):
        self.setWindowTitle('智控系统 - 迷宫路径规划')
        self.setMinimumSize(1200, 900)
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 使用QTabWidget创建多页面
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::tab-bar { alignment: center; }
            QTabBar::tab { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e3f2fd, stop:1 #bbdefb);
                border: 2px solid rgba(44, 90, 160, 0.3);
                border-bottom: none;
                border-radius: 8px 8px 0 0;
                padding: 12px 30px;
                margin: 0 5px;
                font-weight: bold;
                color: #2c5aa0;
                font-size: 14px;
            }
            QTabBar::tab:selected { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #90caf9, stop:1 #64b5f6);
                border-color: rgba(44, 90, 160, 0.6);
            }
            QTabBar::tab:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #bbdefb, stop:1 #90caf9); }
        """)
        
        # ========== 主页面 - 迷宫路径规划 ==========
        main_page = QWidget()
        main_page_layout = QVBoxLayout(main_page)
        main_page_layout.setContentsMargins(20, 20, 20, 20)
        main_page_layout.setSpacing(15)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { background: rgba(44, 90, 160, 0.3); width: 12px; border-radius: 6px; }
            QScrollBar::handle:vertical { background: rgba(44, 90, 160, 0.6); border-radius: 6px; min-height: 30px; }
            QScrollBar::handle:vertical:hover { background: rgba(44, 90, 160, 0.8); }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        """)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(15)
        
        # ========== BLE蓝牙连接区域 ==========
        bt_group = QGroupBox('BLE蓝牙连接')
        bt_group.setStyleSheet(self.get_group_style())
        bt_layout = QHBoxLayout()
        bt_layout.setSpacing(15)
        
        self.combo_bt = QComboBox()
        self.combo_bt.setStyleSheet(self.get_combo_style())
        bt_layout.addWidget(self.combo_bt)
        
        self.btn_refresh = QPushButton('刷新设备')
        self.btn_refresh.setStyleSheet(self.get_button_style('blue'))
        self.btn_refresh.clicked.connect(self.on_refresh_ble)
        bt_layout.addWidget(self.btn_refresh)
        
        self.btn_connect = QPushButton('连接')
        self.btn_connect.setStyleSheet(self.get_button_style('green'))
        self.btn_connect.clicked.connect(self.toggle_ble_connection)
        bt_layout.addWidget(self.btn_connect)
        
        self.lbl_status = QLabel('未连接')
        self.lbl_status.setStyleSheet('color: #c62828; font-weight: bold; font-size: 14px;')
        bt_layout.addWidget(self.lbl_status)
        
        bt_layout.addStretch()
        bt_group.setLayout(bt_layout)
        content_layout.addWidget(bt_group)
        
        # ========== 图像获取区域 ==========
        image_group = QGroupBox('图像获取')
        image_group.setStyleSheet(self.get_group_style())
        image_layout = QHBoxLayout()
        image_layout.setSpacing(15)
        
        cam_layout = QVBoxLayout()
        cam_layout.setSpacing(10)
        
        lbl_cam = QLabel('摄像头ID:')
        lbl_cam.setStyleSheet('color: #2c5aa0; font-weight: 500;')
        cam_layout.addWidget(lbl_cam)
        
        self.spin_cam_id = QSpinBox()
        self.spin_cam_id.setRange(0, 10)
        self.spin_cam_id.setValue(0)
        self.spin_cam_id.setStyleSheet(self.get_spin_style())
        cam_layout.addWidget(self.spin_cam_id)
        
        lbl_resolution = QLabel('分辨率:')
        lbl_resolution.setStyleSheet('color: #2c5aa0; font-weight: 500;')
        cam_layout.addWidget(lbl_resolution)
        
        self.combo_resolution = QComboBox()
        self.combo_resolution.addItems(['360p (640×360)', '720p (1280×720)', '1080p (1920×1080)'])
        self.combo_resolution.setStyleSheet(self.get_combo_style())
        self.combo_resolution.setCurrentIndex(2)  # 默认1080p
        cam_layout.addWidget(self.combo_resolution)
        
        self.btn_open_cam = QPushButton('打开摄像头')
        self.btn_open_cam.setStyleSheet(self.get_button_style('orange'))
        self.btn_open_cam.clicked.connect(self.toggle_camera)
        cam_layout.addWidget(self.btn_open_cam)
        
        self.btn_capture = QPushButton('拍照')
        self.btn_capture.setStyleSheet(self.get_button_style('purple'))
        self.btn_capture.setEnabled(False)
        self.btn_capture.clicked.connect(self.capture_image)
        cam_layout.addWidget(self.btn_capture)
        
        image_layout.addLayout(cam_layout)
        
        self.lbl_cam_preview = QLabel('摄像头预览')
        self.lbl_cam_preview.setStyleSheet('border: 2px solid rgba(44, 90, 160, 0.5); border-radius: 8px; background: rgba(255, 255, 255, 0.7);')
        self.lbl_cam_preview.setMinimumSize(400, 300)
        self.lbl_cam_preview.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.lbl_cam_preview)
        
        file_layout = QVBoxLayout()
        file_layout.setSpacing(10)
        
        self.btn_select_image = QPushButton('选择本地图片')
        self.btn_select_image.setStyleSheet(self.get_button_style('gray'))
        self.btn_select_image.clicked.connect(self.select_local_image)
        file_layout.addWidget(self.btn_select_image)
        
        # 替换为可点击选点的自定义Label
        self.lbl_selected_image = SelectPointLabel('选中的图片')
        self.lbl_selected_image.setStyleSheet('border: 2px solid rgba(44, 90, 160, 0.5); border-radius: 8px; background: rgba(255, 255, 255, 0.7);')
        self.lbl_selected_image.setMinimumSize(400, 300)
        self.lbl_selected_image.setAlignment(Qt.AlignCenter)
        # 绑定选点信号：点击后回填坐标到SpinBox
        self.lbl_selected_image.clicked_point.connect(self.on_image_point_selected)
        file_layout.addWidget(self.lbl_selected_image)
        
        image_layout.addLayout(file_layout)
        image_group.setLayout(image_layout)
        content_layout.addWidget(image_group)
        
        # ========== 箭头检测模式选择 ==========
        arrow_mode_group = QGroupBox('箭头检测模式')
        arrow_mode_group.setStyleSheet(self.get_group_style())
        arrow_mode_layout = QHBoxLayout()
        arrow_mode_layout.setSpacing(20)
        
        self.radio_model = QRadioButton('模型直接检测')
        self.radio_model.setChecked(False)
        self.radio_model.setStyleSheet('color: #2c5aa0;')
        arrow_mode_layout.addWidget(self.radio_model)
        
        self.radio_centroid = QRadioButton('质心检测（推荐）')
        self.radio_centroid.setChecked(True)
        self.radio_centroid.setStyleSheet('color: #2c5aa0;')
        arrow_mode_layout.addWidget(self.radio_centroid)
        
        arrow_mode_group.setLayout(arrow_mode_layout)
        content_layout.addWidget(arrow_mode_group)
        
        # ========== 透视变换角点设置 ==========
        perspective_group = QGroupBox('透视变换角点设置')
        perspective_group.setStyleSheet(self.get_group_style())
        perspective_layout = QHBoxLayout()
        perspective_layout.setSpacing(15)
        
        # 左上角点
        corner_tl_layout = QVBoxLayout()
        corner_tl_layout.setSpacing(5)
        lbl_tl = QLabel('左上角 (X,Y):')
        lbl_tl.setStyleSheet('color: #2c5aa0; font-weight: 500;')
        corner_tl_layout.addWidget(lbl_tl)
        
        tl_h_layout = QHBoxLayout()
        self.spin_tl_x = QSpinBox()
        self.spin_tl_x.setRange(0, 5000)
        self.spin_tl_x.setValue(486)
        self.spin_tl_x.setStyleSheet(self.get_spin_style())
        self.spin_tl_x.setPrefix("X:")
        tl_h_layout.addWidget(self.spin_tl_x)
        
        self.spin_tl_y = QSpinBox()
        self.spin_tl_y.setRange(0, 5000)
        self.spin_tl_y.setValue(42)
        self.spin_tl_y.setStyleSheet(self.get_spin_style())
        self.spin_tl_y.setPrefix("Y:")
        tl_h_layout.addWidget(self.spin_tl_y)
        corner_tl_layout.addLayout(tl_h_layout)
        perspective_layout.addLayout(corner_tl_layout)
        
        # 右上角点
        corner_tr_layout = QVBoxLayout()
        corner_tr_layout.setSpacing(5)
        lbl_tr = QLabel('右上角 (X,Y):')
        lbl_tr.setStyleSheet('color: #2c5aa0; font-weight: 500;')
        corner_tr_layout.addWidget(lbl_tr)
        
        tr_h_layout = QHBoxLayout()
        self.spin_tr_x = QSpinBox()
        self.spin_tr_x.setRange(0, 5000)
        self.spin_tr_x.setValue(1400)
        self.spin_tr_x.setStyleSheet(self.get_spin_style())
        self.spin_tr_x.setPrefix("X:")
        tr_h_layout.addWidget(self.spin_tr_x)
        
        self.spin_tr_y = QSpinBox()
        self.spin_tr_y.setRange(0, 5000)
        self.spin_tr_y.setValue(30)
        self.spin_tr_y.setStyleSheet(self.get_spin_style())
        self.spin_tr_y.setPrefix("Y:")
        tr_h_layout.addWidget(self.spin_tr_y)
        corner_tr_layout.addLayout(tr_h_layout)
        perspective_layout.addLayout(corner_tr_layout)
        
        # 右下角点
        corner_br_layout = QVBoxLayout()
        corner_br_layout.setSpacing(5)
        lbl_br = QLabel('右下角 (X,Y):')
        lbl_br.setStyleSheet('color: #2c5aa0; font-weight: 500;')
        corner_br_layout.addWidget(lbl_br)
        
        br_h_layout = QHBoxLayout()
        self.spin_br_x = QSpinBox()
        self.spin_br_x.setRange(0, 5000)
        self.spin_br_x.setValue(1569)
        self.spin_br_x.setStyleSheet(self.get_spin_style())
        self.spin_br_x.setPrefix("X:")
        br_h_layout.addWidget(self.spin_br_x)
        
        self.spin_br_y = QSpinBox()
        self.spin_br_y.setRange(0, 5000)
        self.spin_br_y.setValue(1043)
        self.spin_br_y.setStyleSheet(self.get_spin_style())
        self.spin_br_y.setPrefix("Y:")
        br_h_layout.addWidget(self.spin_br_y)
        corner_br_layout.addLayout(br_h_layout)
        perspective_layout.addLayout(corner_br_layout)
        
        # 左下角点
        corner_bl_layout = QVBoxLayout()
        corner_bl_layout.setSpacing(5)
        lbl_bl = QLabel('左下角 (X,Y):')
        lbl_bl.setStyleSheet('color: #2c5aa0; font-weight: 500;')
        corner_bl_layout.addWidget(lbl_bl)
        
        bl_h_layout = QHBoxLayout()
        self.spin_bl_x = QSpinBox()
        self.spin_bl_x.setRange(0, 5000)
        self.spin_bl_x.setValue(281)
        self.spin_bl_x.setStyleSheet(self.get_spin_style())
        self.spin_bl_x.setPrefix("X:")
        bl_h_layout.addWidget(self.spin_bl_x)
        
        self.spin_bl_y = QSpinBox()
        self.spin_bl_y.setRange(0, 5000)
        self.spin_bl_y.setValue(1017)
        self.spin_bl_y.setStyleSheet(self.get_spin_style())
        self.spin_bl_y.setPrefix("Y:")
        bl_h_layout.addWidget(self.spin_bl_y)
        corner_bl_layout.addLayout(bl_h_layout)
        perspective_layout.addLayout(corner_bl_layout)
        
        perspective_group.setLayout(perspective_layout)
        content_layout.addWidget(perspective_group)
        
        # ========== YOLO检测区域 ==========
        yolo_group = QGroupBox('YOLO目标检测')
        yolo_group.setStyleSheet(self.get_group_style())
        yolo_layout = QHBoxLayout()
        yolo_layout.setSpacing(15)
        
        model_layout = QVBoxLayout()
        model_layout.setSpacing(10)
        
        lbl_model = QLabel('YOLO模型文件:')
        lbl_model.setStyleSheet('color: #2c5aa0; font-weight: 500;')
        model_layout.addWidget(lbl_model)
        
        self.line_model = QTextEdit()
        self.line_model.setFixedHeight(30)
        self.line_model.setStyleSheet('border: 2px solid rgba(44, 90, 160, 0.4); border-radius: 8px; padding: 4px;')
        model_layout.addWidget(self.line_model)
        
        self.btn_select_model = QPushButton('选择模型')
        self.btn_select_model.setStyleSheet(self.get_button_style('blue'))
        self.btn_select_model.clicked.connect(self.select_model_file)
        model_layout.addWidget(self.btn_select_model)
        
        self.btn_detect = QPushButton('开始检测')
        self.btn_detect.setStyleSheet(self.get_button_style('green'))
        self.btn_detect.clicked.connect(self.run_detection)
        model_layout.addWidget(self.btn_detect)
        
        yolo_layout.addLayout(model_layout)
        
        result_layout = QVBoxLayout()
        result_layout.setSpacing(10)
        
        lbl_result = QLabel('检测结果:')
        lbl_result.setStyleSheet('color: #2c5aa0; font-weight: 500;')
        result_layout.addWidget(lbl_result)
        
        self.txt_detection_result = QTextEdit()
        self.txt_detection_result.setFixedHeight(100)
        self.txt_detection_result.setReadOnly(True)
        self.txt_detection_result.setStyleSheet('border: 2px solid rgba(44, 90, 160, 0.4); border-radius: 8px; padding: 8px;')
        result_layout.addWidget(self.txt_detection_result)
        
        yolo_layout.addLayout(result_layout)
        yolo_group.setLayout(yolo_layout)
        content_layout.addWidget(yolo_group)
        
        # ========== 迷宫显示区域 ==========
        maze_group = QGroupBox('迷宫显示')
        maze_group.setStyleSheet(self.get_group_style())
        maze_layout = QVBoxLayout()
        maze_layout.setSpacing(10)
        
        self.lbl_start_info = QLabel('请先进行图像检测，然后点击迷宫边界格子设置起点')
        self.lbl_start_info.setStyleSheet('color: #2c5aa0; font-weight: 500;')
        maze_layout.addWidget(self.lbl_start_info)
        
        self.maze_grid = MazeGridWidget()
        self.maze_grid.setFixedSize(420, 420)
        self.maze_grid.setStyleSheet('border: 2px solid rgba(44, 90, 160, 0.5); border-radius: 8px;')
        self.maze_grid.cell_clicked.connect(self.on_cell_clicked)
        maze_layout.addWidget(self.maze_grid, 0, Qt.AlignCenter)
        
        maze_group.setLayout(maze_layout)
        content_layout.addWidget(maze_group)
        
        # ========== 路径规划区域 ==========
        path_group = QGroupBox('路径规划')
        path_group.setStyleSheet(self.get_group_style())
        path_layout = QVBoxLayout()
        path_layout.setSpacing(10)
        
        self.btn_plan = QPushButton('规划路径')
        self.btn_plan.setStyleSheet(self.get_button_style('blue'))
        self.btn_plan.setEnabled(False)
        self.btn_plan.clicked.connect(self.plan_path)
        path_layout.addWidget(self.btn_plan)
        
        self.list_path = QListWidget()
        self.list_path.setStyleSheet('border: 2px solid rgba(44, 90, 160, 0.4); border-radius: 8px; padding: 4px;')
        path_layout.addWidget(self.list_path)
        
        send_layout = QHBoxLayout()
        
        self.btn_send = QPushButton('发送全部指令')
        self.btn_send.setStyleSheet(self.get_button_style('green'))
        self.btn_send.setEnabled(False)
        self.btn_send.clicked.connect(self.send_all_commands)
        send_layout.addWidget(self.btn_send)
        
        self.btn_return = QPushButton('返回起点')
        self.btn_return.setStyleSheet(self.get_button_style('orange'))
        self.btn_return.setEnabled(False)
        self.btn_return.clicked.connect(self.generate_return_path)
        send_layout.addWidget(self.btn_return)
        
        send_layout.addStretch()
        path_layout.addLayout(send_layout)
        
        path_group.setLayout(path_layout)
        content_layout.addWidget(path_group)
        
        content_layout.addStretch()
        
        scroll_area.setWidget(content_widget)
        main_page_layout.addWidget(scroll_area)
        
        self.tab_widget.addTab(main_page, '迷宫路径规划')
        
        # ========== 测试区页面 ==========
        self.test_page = QWidget()
        self.init_test_page()
        self.tab_widget.addTab(self.test_page, '蓝牙测试')
        
        main_layout.addWidget(self.tab_widget)
        self.setLayout(main_layout)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_frame)
        
    def get_group_style(self):
        return """
            QGroupBox { font-weight: bold; font-size: 14px; color: #2c5aa0;
                border: 2px solid rgba(44, 90, 160, 0.5); border-radius: 8px;
                margin-top: 10px; padding-top: 10px; background: rgba(255, 255, 255, 0.8); }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        """
        
    def get_button_style(self, color):
        colors = {
            'blue': 'background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a90e2, stop:1 #357abd); color: white;',
            'green': 'background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5cb85c, stop:1 #4cae4c); color: white;',
            'orange': 'background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f0ad4e, stop:1 #ec971f); color: white;',
            'purple': 'background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #9b59b6, stop:1 #8e44ad); color: white;',
            'gray': 'background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #95a5a6, stop:1 #7f8c8d); color: white;'
        }
        return f"""
            QPushButton {{ {colors.get(color, colors['gray'])} border: none; border-radius: 6px;
                padding: 8px 16px; font-weight: bold; font-size: 13px; }}
            QPushButton:hover {{ opacity: 0.9; }}
            QPushButton:pressed {{ opacity: 0.8; }}
            QPushButton:disabled {{ background: #cccccc; }}
        """
        
    def get_combo_style(self):
        return """
            QComboBox { border: 2px solid rgba(44, 90, 160, 0.4); border-radius: 6px;
                padding: 5px 10px; background: rgba(255, 255, 255, 0.9); min-width: 200px; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox::down-arrow { width: 10px; height: 10px; }
        """
        
    def get_spin_style(self):
        return """
            QSpinBox { border: 2px solid rgba(44, 90, 160, 0.4); border-radius: 6px;
                padding: 5px 8px; background: rgba(255, 255, 255, 0.9); }
        """
        
    # ========== BLE蓝牙功能 ==========
    def on_refresh_ble(self):
        """刷新BLE设备（使用工作线程）"""
        self.btn_refresh.setEnabled(False)
        self.combo_bt.clear()
        self.btn_refresh.setEnabled(False)
        self.lbl_status.setText('搜索中...')
        
        # 使用传统蓝牙扫描
        devices = self.bluetooth_manager.discover_devices()
        self.ble_devices = devices
        
        for name, addr in devices:
            display_name = f"{name} ({addr})" if name else f"未知设备 ({addr})"
            self.combo_bt.addItem(display_name)
        
        self.btn_refresh.setEnabled(True)
        if not devices:
            self.combo_bt.addItem('未发现蓝牙设备')
            self.lbl_status.setText('未连接')
            self.lbl_status.setStyleSheet('color: #c62828; font-weight: bold; font-size: 14px;')
            
    def toggle_ble_connection(self):
        if self.bluetooth_manager.is_connected():
            self.btn_connect.setEnabled(False)
            self.bluetooth_manager.disconnect_device()
            self.btn_connect.setEnabled(True)
            self.lbl_status.setText('未连接')
            self.lbl_status.setStyleSheet('color: #c62828; font-weight: bold; font-size: 14px;')
            self.btn_connect.setText('连接')
        else:
            if self.combo_bt.count() == 0:
                QMessageBox.warning(self, '警告', '请先刷新蓝牙设备')
                return
            index = self.combo_bt.currentIndex()
            if index < 0 or index >= len(self.ble_devices):
                QMessageBox.warning(self, '警告', '请选择有效的蓝牙设备')
                return
            addr = self.ble_devices[index][1]
            self.btn_connect.setEnabled(False)
            
            success, error_msg = self.bluetooth_manager.connect_device(addr)
            
            self.btn_connect.setEnabled(True)
            if success:
                self.lbl_status.setText('已连接')
                self.lbl_status.setStyleSheet('color: #4cae4c; font-weight: bold; font-size: 14px;')
                self.btn_connect.setText('断开')
            else:
                QMessageBox.critical(self, '错误', f'蓝牙连接失败: {error_msg}')
                
    # ========== 测试区页面初始化 ==========
    def init_test_page(self):
        """初始化蓝牙测试页面"""
        main_layout = QVBoxLayout(self.test_page)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # ========== 方向指令合成区域 ==========
        cmd_group = QGroupBox('方向指令合成')
        cmd_group.setStyleSheet(self.get_group_style())
        cmd_layout = QGridLayout()
        cmd_layout.setSpacing(15)
        
        lbl_dir = QLabel('方向:')
        lbl_dir.setStyleSheet('color: #2c5aa0; font-weight: 500;')
        cmd_layout.addWidget(lbl_dir, 0, 0)
        
        self.combo_test_dir = QComboBox()
        self.combo_test_dir.addItems(['FORWARD', 'BACKWARD', 'LEFT', 'RIGHT'])
        self.combo_test_dir.setMinimumWidth(120)
        self.combo_test_dir.setStyleSheet(self.get_combo_style())
        cmd_layout.addWidget(self.combo_test_dir, 0, 1)
        
        lbl_speed = QLabel('速度 (1-8):')
        lbl_speed.setStyleSheet('color: #2c5aa0; font-weight: 500;')
        cmd_layout.addWidget(lbl_speed, 0, 2)
        
        self.spin_test_speed = QSpinBox()
        self.spin_test_speed.setRange(1, 8)
        self.spin_test_speed.setValue(4)
        self.spin_test_speed.setStyleSheet(self.get_spin_style())
        cmd_layout.addWidget(self.spin_test_speed, 0, 3)
        
        self.btn_send_dir = QPushButton('发送方向指令')
        self.btn_send_dir.setStyleSheet(self.get_button_style('orange'))
        self.btn_send_dir.clicked.connect(self.send_test_direction_cmd)
        cmd_layout.addWidget(self.btn_send_dir, 0, 4)
        
        self.lbl_dir_preview = QLabel('预览: FORWARD 4')
        self.lbl_dir_preview.setStyleSheet('color: #666; font-style: italic; font-size: 13px;')
        cmd_layout.addWidget(self.lbl_dir_preview, 1, 0, 1, 5)
        
        self.combo_test_dir.currentTextChanged.connect(self.update_test_preview)
        self.spin_test_speed.valueChanged.connect(self.update_test_preview)
        
        cmd_group.setLayout(cmd_layout)
        main_layout.addWidget(cmd_group)
        
        # ========== STOP 和 RUN 区域 ==========
        ctrl_group = QGroupBox('控制指令')
        ctrl_group.setStyleSheet(self.get_group_style())
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(30)
        
        # STOP 按钮
        self.btn_test_stop = QPushButton('STOP')
        self.btn_test_stop.setMinimumHeight(60)
        self.btn_test_stop.setMinimumWidth(140)
        self.btn_test_stop.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ef5350, stop:1 #e53935);
                color: white;
                font-size: 16pt;
                font-weight: bold;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #e53935, stop:1 #c62828); }
            QPushButton:pressed { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #c62828, stop:1 #b71c1c); }
        """)
        self.btn_test_stop.clicked.connect(self.send_test_stop)
        ctrl_layout.addWidget(self.btn_test_stop)
        
        # RUN 区域
        run_layout = QVBoxLayout()
        run_layout.setSpacing(10)
        
        run_row = QHBoxLayout()
        run_row.setSpacing(12)
        
        lbl_run = QLabel('RUN 值 (-100~100):')
        lbl_run.setStyleSheet('color: #2c5aa0; font-weight: 500;')
        run_row.addWidget(lbl_run)
        
        self.slider_run = QSlider(Qt.Horizontal)
        self.slider_run.setRange(-100, 100)
        self.slider_run.setValue(50)
        self.slider_run.setStyleSheet("""
            QSlider::groove:horizontal { height: 8px; background: rgba(44, 90, 160, 0.2); border-radius: 4px; }
            QSlider::handle:horizontal { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #64b5f6, stop:1 #42a5f5); width: 20px; height: 20px; border-radius: 50%; margin: -6px 0; }
            QSlider::sub-page:horizontal { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #64b5f6, stop:1 #42a5f5); border-radius: 4px; }
        """)
        run_row.addWidget(self.slider_run)
        
        self.lbl_run_val = QLabel('50')
        self.lbl_run_val.setMinimumWidth(40)
        self.lbl_run_val.setStyleSheet('color: #2c5aa0; font-weight: bold; font-size: 14px; text-align: center;')
        run_row.addWidget(self.lbl_run_val)
        self.slider_run.valueChanged.connect(self.update_run_label)
        run_layout.addLayout(run_row)
        
        self.btn_send_run = QPushButton('发送 RUN 指令')
        self.btn_send_run.setStyleSheet(self.get_button_style('green'))
        self.btn_send_run.clicked.connect(self.send_test_run)
        run_layout.addWidget(self.btn_send_run)
        
        ctrl_layout.addLayout(run_layout)
        ctrl_group.setLayout(ctrl_layout)
        main_layout.addWidget(ctrl_group)
        
        # ========== 自定义指令数组发送 ==========
        custom_group = QGroupBox('自定义指令数组发送')
        custom_group.setStyleSheet(self.get_group_style())
        custom_layout = QVBoxLayout()
        custom_layout.setSpacing(10)
        
        lbl_custom = QLabel('指令列表 (每行一条指令，格式: 方向 数字)')
        lbl_custom.setStyleSheet('color: #2c5aa0; font-weight: 500;')
        custom_layout.addWidget(lbl_custom)
        
        # 快捷指令按钮
        quick_layout = QHBoxLayout()
        quick_layout.setSpacing(8)
        
        btn_forward = QPushButton('FORWARD')
        btn_forward.setStyleSheet(self.get_button_style('blue'))
        btn_forward.setFixedWidth(90)
        btn_forward.clicked.connect(lambda: self.add_quick_cmd('FORWARD'))
        quick_layout.addWidget(btn_forward)
        
        btn_backward = QPushButton('BACKWARD')
        btn_backward.setStyleSheet(self.get_button_style('blue'))
        btn_backward.setFixedWidth(90)
        btn_backward.clicked.connect(lambda: self.add_quick_cmd('BACKWARD'))
        quick_layout.addWidget(btn_backward)
        
        btn_left = QPushButton('LEFT')
        btn_left.setStyleSheet(self.get_button_style('blue'))
        btn_left.setFixedWidth(90)
        btn_left.clicked.connect(lambda: self.add_quick_cmd('LEFT'))
        quick_layout.addWidget(btn_left)
        
        btn_right = QPushButton('RIGHT')
        btn_right.setStyleSheet(self.get_button_style('blue'))
        btn_right.setFixedWidth(90)
        btn_right.clicked.connect(lambda: self.add_quick_cmd('RIGHT'))
        quick_layout.addWidget(btn_right)
        
        quick_layout.addStretch()
        custom_layout.addLayout(quick_layout)
        
        # 速度选择
        speed_layout = QHBoxLayout()
        speed_layout.setSpacing(8)
        
        lbl_speed = QLabel('选择速度:')
        lbl_speed.setStyleSheet('color: #2c5aa0; font-weight: 500;')
        speed_layout.addWidget(lbl_speed)
        
        self.combo_speed = QComboBox()
        self.combo_speed.addItems(['1', '2', '3', '4', '5', '6', '7', '8'])
        self.combo_speed.setCurrentText('4')
        self.combo_speed.setStyleSheet(self.get_combo_style())
        self.combo_speed.setFixedWidth(80)
        speed_layout.addWidget(self.combo_speed)
        
        speed_layout.addStretch()
        custom_layout.addLayout(speed_layout)
        
        self.txt_custom_cmds = QTextEdit()
        self.txt_custom_cmds.setFixedHeight(80)
        self.txt_custom_cmds.setPlaceholderText('使用上方快捷按钮添加指令，或从规划路径导入')
        self.txt_custom_cmds.setStyleSheet('border: 2px solid rgba(44, 90, 160, 0.4); border-radius: 8px; padding: 8px;')
        self.txt_custom_cmds.textChanged.connect(self.update_custom_preview)
        custom_layout.addWidget(self.txt_custom_cmds)
        
        # 操作按钮
        action_layout = QHBoxLayout()
        action_layout.setSpacing(10)
        
        self.btn_import_path = QPushButton('从规划路径导入')
        self.btn_import_path.setStyleSheet(self.get_button_style('green'))
        self.btn_import_path.clicked.connect(self.import_from_path)
        action_layout.addWidget(self.btn_import_path)
        
        self.btn_clear_cmds = QPushButton('清空指令')
        self.btn_clear_cmds.setStyleSheet(self.get_button_style('gray'))
        self.btn_clear_cmds.clicked.connect(self.clear_custom_cmds)
        action_layout.addWidget(self.btn_clear_cmds)
        
        self.btn_send_custom = QPushButton('发送自定义指令数组')
        self.btn_send_custom.setStyleSheet(self.get_button_style('purple'))
        self.btn_send_custom.clicked.connect(self.send_custom_commands)
        action_layout.addWidget(self.btn_send_custom)
        
        custom_layout.addLayout(action_layout)
        
        self.lbl_custom_preview = QLabel('发送格式预览: {指令一，指令二，指令三}')
        self.lbl_custom_preview.setStyleSheet('color: #666; font-style: italic; font-size: 12px;')
        custom_layout.addWidget(self.lbl_custom_preview)
        
        custom_group.setLayout(custom_layout)
        main_layout.addWidget(custom_group)
        
        # ========== 发送记录区域 ==========
        log_group = QGroupBox('发送记录')
        log_group.setStyleSheet(self.get_group_style())
        log_layout = QVBoxLayout()
        log_layout.setSpacing(10)
        
        self.list_test_log = QListWidget()
        self.list_test_log.setStyleSheet('border: 2px solid rgba(44, 90, 160, 0.3); border-radius: 8px; background: rgba(248, 252, 255, 0.9); font-family: monospace; font-size: 12px;')
        log_layout.addWidget(self.list_test_log)
        
        btn_clear_log = QPushButton('清空记录')
        btn_clear_log.setStyleSheet(self.get_button_style('gray'))
        btn_clear_log.clicked.connect(self.list_test_log.clear)
        log_layout.addWidget(btn_clear_log)
        
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        main_layout.addStretch()
    
    # ========== 测试区功能方法 ==========
    def update_test_preview(self):
        d = self.combo_test_dir.currentText()
        s = self.spin_test_speed.value()
        self.lbl_dir_preview.setText(f'预览: {d} {s}')
    
    def update_run_label(self):
        val = self.slider_run.value()
        self.lbl_run_val.setText(str(val))
    
    def add_quick_cmd(self, direction):
        """添加快捷指令"""
        speed = self.combo_speed.currentText()
        cmd = f'{direction} {speed}'
        
        # 如果已经有文本，添加换行
        current_text = self.txt_custom_cmds.toPlainText()
        if current_text.strip():
            self.txt_custom_cmds.append(cmd)
        else:
            self.txt_custom_cmds.setPlainText(cmd)
        
        # 更新预览
        self.update_custom_preview()
    
    def import_from_path(self):
        """从规划路径导入"""
        if not self.commands:
            QMessageBox.warning(self, '警告', '主页面暂无规划路径，请先进行路径规划')
            return
            
        self.txt_custom_cmds.clear()
        for cmd in self.commands:
            self.txt_custom_cmds.append(cmd)
        
        self.update_custom_preview()
        QMessageBox.information(self, '成功', f'已导入 {len(self.commands)} 条指令')
    
    def clear_custom_cmds(self):
        """清空自定义指令"""
        self.txt_custom_cmds.clear()
        self.update_custom_preview()
    
    def update_custom_preview(self):
        """更新自定义指令预览"""
        text = self.txt_custom_cmds.toPlainText().strip()
        if text:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            if lines:
                preview = '{' + ','.join(lines) + '}'
                self.lbl_custom_preview.setText(f'发送格式预览: {preview}')
            else:
                self.lbl_custom_preview.setText('发送格式预览: {指令一,指令二,指令三}')
        else:
            self.lbl_custom_preview.setText('发送格式预览: {指令一,指令二,指令三}')
    
    def send_test_command(self, cmd):
        """发送单个测试指令"""
        self.list_test_log.addItem(f'[发送] {cmd}')
        self.list_test_log.scrollToBottom()
        
        if not self.bluetooth_manager.is_connected():
            self.list_test_log.addItem('[警告] 蓝牙未连接，仅记录指令')
            self.list_test_log.scrollToBottom()
            return
            
        try:
            success, error_msg = self.bluetooth_manager.send_data(cmd)
            if success:
                self.list_test_log.addItem(f'[成功] {cmd} 已发送')
            else:
                self.list_test_log.addItem(f'[失败] {error_msg}')
        except Exception as e:
            self.list_test_log.addItem(f'[失败] {str(e)}')
        self.list_test_log.scrollToBottom()
    
    def send_test_direction_cmd(self):
        d = self.combo_test_dir.currentText()
        s = self.spin_test_speed.value()
        cmd = f'{d} {s}'
        self.send_test_command(cmd)
    
    def send_test_stop(self):
        self.send_test_command('STOP')
    
    def send_test_run(self):
        val = self.slider_run.value()
        self.lbl_run_val.setText(str(val))
        cmd = f'RUN {val}'
        self.send_test_command(cmd)
    
    def send_custom_commands(self):
        """发送自定义指令数组"""
        text = self.txt_custom_cmds.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, '警告', '请输入指令')
            return
            
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            QMessageBox.warning(self, '警告', '请输入有效的指令')
            return
            
        # 生成发送格式: {指令一，指令二，指令三}
        cmd_str = "{" + ",".join(lines) + "}"
        self.list_test_log.addItem(f'[发送数组] {cmd_str}')
        self.list_test_log.scrollToBottom()
        
        if not self.bluetooth_manager.is_connected():
            self.list_test_log.addItem('[警告] 蓝牙未连接，仅记录指令')
            self.list_test_log.scrollToBottom()
            return
            
        try:
            success, error_msg = self.bluetooth_manager.send_data(cmd_str)
            if success:
                self.list_test_log.addItem(f'[成功] 指令数组已发送')
            else:
                self.list_test_log.addItem(f'[失败] {error_msg}')
        except Exception as e:
            self.list_test_log.addItem(f'[失败] {str(e)}')
        self.list_test_log.scrollToBottom()
                
    # ========== 摄像头功能 ==========
    def toggle_camera(self):
        if self.cap and self.cap.isOpened():
            self.close_camera()
        else:
            self.open_camera()
            
    def open_camera(self):
        cam_id = self.spin_cam_id.value()
        self.cap = cv2.VideoCapture(cam_id)
        
        resolution_map = {
            0: (640, 360),
            1: (1280, 720),
            2: (1920, 1080)
        }
        width, height = resolution_map[self.combo_resolution.currentIndex()]
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if self.cap.isOpened():
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"摄像头已打开，分辨率: {actual_width}×{actual_height}")
            
            self.timer.start(30)
            self.btn_open_cam.setText('关闭摄像头')
            self.btn_capture.setEnabled(True)
        else:
            QMessageBox.warning(self, '警告', '摄像头无法打开')
            
    def close_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.lbl_cam_preview.clear()
        self.btn_open_cam.setText('打开摄像头')
        self.btn_capture.setEnabled(False)
        
    def update_camera_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.lbl_cam_preview.setPixmap(QPixmap.fromImage(q_img).scaled(
                    self.lbl_cam_preview.width(), self.lbl_cam_preview.height(), Qt.KeepAspectRatio))
                    
    def capture_image(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.detected_image = frame.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.lbl_selected_image.setPixmap(QPixmap.fromImage(q_img).scaled(
                    self.lbl_selected_image.width(), self.lbl_selected_image.height(), Qt.KeepAspectRatio))
                self.lbl_selected_image.set_origin_image(self.detected_image)


    def select_local_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '选择图片', '', '图像文件 (*.jpg *.jpeg *.png *.bmp)')
        if file_path:
            self.detected_image = cv2.imread(file_path)
            if self.detected_image is not None:
                frame_rgb = cv2.cvtColor(self.detected_image, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.lbl_selected_image.setPixmap(QPixmap.fromImage(q_img).scaled(
                    self.lbl_selected_image.width(), self.lbl_selected_image.height(), Qt.KeepAspectRatio))
                self.lbl_selected_image.set_origin_image(self.detected_image)

    # ========== YOLO模型选择 ==========
    def select_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '选择模型', '', '模型文件 (*.pt *.pth)')
        if file_path:
            self.line_model.setText(file_path)
            try:
                self.yolo_model = YOLO(file_path)
                QMessageBox.information(self, '成功', 'YOLO模型加载成功')
            except Exception as e:
                QMessageBox.critical(self, '错误', f'模型加载失败: {str(e)}')
                
    # ========== 视觉检测（seal.py逻辑） ==========
    def run_detection(self):
        if self.detected_image is None:
            QMessageBox.warning(self, '错误', '请先拍照或选择图片')
            return
            
        if self.yolo_model is None:
            QMessageBox.warning(self, '错误', '请先选择YOLO模型')
            return
            
        try:
            img_h, img_w = self.detected_image.shape[:2]
            print(f"原始图像尺寸: {img_w} x {img_h}")
            
            # 获取用户设置的四个角点
            src_corners = [
                [self.spin_tl_x.value(), self.spin_tl_y.value()],  # 左上
                [self.spin_tr_x.value(), self.spin_tr_y.value()],  # 右上
                [self.spin_br_x.value(), self.spin_br_y.value()],  # 右下
                [self.spin_bl_x.value(), self.spin_bl_y.value()]   # 左下
            ]
            print(f"透视变换角点: {src_corners}")
            
            # 进行透视变换
            transformed_img = perspective_transform(self.detected_image, src_corners)
            print(f"透视变换后图像尺寸: {transformed_img.shape[1]} x {transformed_img.shape[0]}")
            
            # 使用seal.py的get_lines方法切分格子
            cells = get_lines(transformed_img)
            
            if not cells:
                QMessageBox.warning(self, '警告', '未能检测到网格线')
                return
                
            # 初始化7×7迷宫
            self.maze_data = [[{'number': None, 'direction': None} for _ in range(7)] for _ in range(7)]
            
            # 获取箭头检测模式
            use_centroid = self.radio_centroid.isChecked()
            
            result_text = "检测结果:\n"
            detected_count = 0
            
            # 处理每个格子
            for idx, cell in enumerate(cells):
                row = idx // 7
                col = idx % 7
                
                if row < 7 and col < 7:
                    results = divide_dect(cell, self.yolo_model, use_centroid)
                    
                    if results:
                        for res in results:
                            direction = res.get('direction')
                            number = res.get('number')
                            
                            if direction:
                                self.maze_data[row][col]['direction'] = direction
                                result_text += f"格子({row},{col}): 方向={direction}\n"
                                detected_count += 1
                            if number is not None:
                                self.maze_data[row][col]['number'] = number
                                result_text += f"格子({row},{col}): 数字={number}\n"
                                detected_count += 1
            
            if detected_count == 0:
                result_text = "警告: 未检测到任何目标!"
                QMessageBox.warning(self, '检测结果', result_text)
            else:
                QMessageBox.information(self, '成功', f'检测完成! 共检测到 {detected_count} 个目标')
            
            self.txt_detection_result.setText(result_text)
            self.maze_grid.set_maze_data(self.maze_data)
            
            self.btn_plan.setEnabled(True)
            # 终点固定在左上角(0,0)，标记终点位置
            self.maze_grid.set_start_point(0, 0)
            self.lbl_start_info.setText('终点固定在左上角(0,0)，请点击迷宫边界格子设置起点')
            
        except Exception as e:
            error_msg = f'检测失败: {str(e)}'
            QMessageBox.critical(self, '检测失败', error_msg)
            print(f"检测错误: {str(e)}")
            
    # ========== 迷宫路径规划 ==========
    def on_cell_clicked(self, row, col):
        if self.maze_data is None:
            return
        
        # 终点固定在左上角(0,0)，不能作为起点
        if row == 0 and col == 0:
            QMessageBox.warning(self, '提示', '左上角(0,0)是终点，请选择其他边界格子作为起点')
            return
            
        # 只能选择边界格子作为起点
        if row == 0 or row == 6 or col == 0 or col == 6:
            self.start_point = (row, col)
            self.maze_grid.set_start_point(row, col)
            self.lbl_start_info.setText(f'起点已设置: ({row},{col})，终点固定在(0,0)')
            
    def plan_path(self):
        if self.maze_data is None or self.start_point is None:
            return
            
        from collections import deque
        
        visited = [[False for _ in range(7)] for _ in range(7)]
        queue = deque()
        queue.append((self.start_point[0], self.start_point[1], []))
        visited[self.start_point[0]][self.start_point[1]] = True
        
        # 终点固定在左上角(0,0)
        target_pos = (0, 0)
        found_path = None
        
        directions = {
            'FORWARD': (-1, 0),
            'BACKWARD': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }
        
        while queue:
            row, col, path = queue.popleft()
            
            if (row, col) == target_pos:
                found_path = path.copy()
                break
                
            cell = self.maze_data[row][col]
            direction = cell.get('direction')
            number = cell.get('number')
            
            if direction and direction in directions and number is not None and number > 0:
                dr, dc = directions[direction]
                new_row = row + dr * number
                new_col = col + dc * number
                
                if 0 <= new_row < 7 and 0 <= new_col < 7 and not visited[new_row][new_col]:
                    visited[new_row][new_col] = True
                    new_path = path.copy()
                    new_path.append((new_row, new_col, f'{direction} {number}'))
                    queue.append((new_row, new_col, new_path))
                    
        if found_path:
            self.path = found_path
            self.path_cells = [(p[0], p[1]) for p in found_path]
            self.maze_grid.set_path_cells(self.path_cells)
            self.maze_grid.set_start_point(self.start_point[0], self.start_point[1])
            
            self.commands = [p[2] for p in found_path]
            self.list_path.clear()
            for cmd in self.commands:
                self.list_path.addItem(cmd)
                
            # 只有路径完整到达终点才启用发送按钮
            self.btn_send.setEnabled(True)
            self.btn_return.setEnabled(False)
            QMessageBox.information(self, '成功', f'路径规划完成! 共{len(self.commands)}步到达终点(0,0)')
        else:
            # 路径规划失败，不启用发送按钮
            self.btn_send.setEnabled(False)
            self.btn_return.setEnabled(False)
            self.commands = []
            self.list_path.clear()
            QMessageBox.warning(self, '路径规划失败', '无法找到从起点到达终点(0,0)的完整路径，请检查迷宫数据是否完整')
            
    def generate_return_path(self):
        if not self.commands:
            return
            
        reverse_map = {
            'FORWARD': 'BACKWARD',
            'BACKWARD': 'FORWARD',
            'LEFT': 'RIGHT',
            'RIGHT': 'LEFT'
        }
        
        return_commands = []
        for cmd in reversed(self.commands):
            direction, number = cmd.split()
            reverse_dir = reverse_map[direction]
            return_commands.append(f'{reverse_dir} {number}')
            
        self.return_commands = return_commands
        self.list_path.clear()
        for cmd in self.return_commands:
            self.list_path.addItem(cmd)
            
        self.commands = return_commands
        self.btn_return.setEnabled(False)
        QMessageBox.information(self, '成功', '已生成返回起点的路径!')
            
    def send_all_commands(self):
        if not self.commands:
            return
            
        if not self.bluetooth_manager.is_connected():
            QMessageBox.warning(self, '警告', '请先连接蓝牙')
            return
            
        # 生成发送格式: {指令一，指令二}
        cmd_str = "{" + ",".join(self.commands) + "}"
        print(f"发送指令: {cmd_str}")
        
        try:
            success, error_msg = self.bluetooth_manager.send_data(cmd_str)
            
            if success:
                QMessageBox.information(self, '成功', '指令发送完成!')
                
                if not self.return_commands:
                    self.btn_return.setEnabled(True)
            else:
                QMessageBox.critical(self, '错误', f'指令发送失败: {error_msg}')
                
        except Exception as e:
            QMessageBox.critical(self, '错误', f'指令发送失败: {str(e)}')
    def closeEvent(self, event):
        self.close_camera()
        self.bluetooth_manager.disconnect_device()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FinalApp()
    window.show()
    sys.exit(app.exec_())
