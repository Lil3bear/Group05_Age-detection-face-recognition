# 导入必要的库
import os
import sys
import cv2
import time
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from insightface.app import FaceAnalysis
from collections import deque

def parse_args():
    parser = argparse.ArgumentParser(description='人脸性别年龄检测')
    parser.add_argument('--image', type=str, help='图片路径，不指定则使用摄像头')
    parser.add_argument('--camera', type=int, default=0, help='摄像头设备号，默认0')
    parser.add_argument('--output', type=str, help='输出图片路径，仅在处理单张图片时有效')
    parser.add_argument('--width', type=int, default=640, help='处理分辨率宽度，默认640')
    parser.add_argument('--height', type=int, default=480, help='处理分辨率高度，默认480')
    parser.add_argument('--interval', type=int, default=1, help='处理间隔帧数，默认1')
    return parser.parse_args()

# 尝试检测是否有可用的 CUDA 提供者，否则回退到 CPU
try:
    import onnxruntime as ort
    available = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available:
        providers = ['CUDAExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
except Exception:
    # 无法导入 onnxruntime 或检测失败，使用 CPU 以保证最大兼容性
    providers = ['CPUExecutionProvider']

# 1. 初始化模型（自动下载预训练权重，首次运行会慢一点，需联网）
# providers 参数将根据运行环境自动选择
print(f"使用的 ONNX 提供者: {providers}")
app = FaceAnalysis(providers=providers)

# 2. 准备模型（设置检测图像尺寸，640x640适合大部分场景）
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 表示第一个设备（GPU 或 CPU）

# 加载中文字体
def load_font():
    try:
        # 尝试使用更纱黑体（如果安装了的话）
        font_path = "C:/Windows/Fonts/sarasa-mono-sc-regular.ttf"
        font_small = ImageFont.truetype(font_path, 40)  # 摄像头模式字体稍小
        font_large = ImageFont.truetype(font_path, 60)
    except:
        # 回退到微软雅黑
        font_path = "C:/Windows/Fonts/msyh.ttc"
        font_small = ImageFont.truetype(font_path, 40)
        font_large = ImageFont.truetype(font_path, 60)
    return font_small, font_large

class FPSCounter:
    def __init__(self, window_size=30):
        self.frame_times = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        self.last_time = time.time()
        
    def update(self, processing_time=None):
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        if processing_time is not None:
            self.processing_times.append(processing_time)
        self.last_time = current_time
        
    def get_fps(self):
        if not self.frame_times:
            return 0
        return len(self.frame_times) / sum(self.frame_times)
        
    def get_processing_time(self):
        if not self.processing_times:
            return 0
        return sum(self.processing_times) / len(self.processing_times) * 1000  # 转换为毫秒

def process_frame(frame, app, font_small, font_large, processing_scale=1.0):
    # 记录处理开始时间
    start_time = time.time()
    
    # 如果需要缩放处理
    if processing_scale != 1.0:
        process_frame = cv2.resize(frame, None, fx=processing_scale, fy=processing_scale)
    else:
        process_frame = frame
    
    # 执行人脸分析（检测人脸+预测年龄/性别）
    faces = app.get(process_frame)
    
    # 如果进行了缩放处理，需要将检测结果坐标还原
    if processing_scale != 1.0:
        scale = 1.0 / processing_scale
        for face in faces:
            face.bbox *= scale

    # 处理结果并显示
    for i, face in enumerate(faces):
        # 人脸位置（左上角x,y，右下角x,y）
        bbox = face.bbox.astype(int)
        age = int(face.age)
        gender = "男" if face.gender == 1 else "女"

        # 画框
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # 转换为PIL以绘制中文
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        # 文字位置计算
        y_text = bbox[1] - 70
        if y_text < 20:
            y_text = bbox[3] + 30

        # 绘制性别
        gender_text = f"{gender},"
        gender_bbox = draw.textbbox((0, 0), gender_text, font=font_small)
        gender_width = gender_bbox[2] - gender_bbox[0]
        draw.text((bbox[0], y_text), gender_text, fill=(255, 0, 0), font=font_small)

        # 绘制年龄
        age_text = f"{age}岁"
        draw.text((bbox[0] + gender_width + 10, y_text - 10), age_text, fill=(255, 0, 0), font=font_large)

        # 转回OpenCV格式
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # 计算处理时间
    process_time = time.time() - start_time
    return frame, process_time

def create_window(window_name, width=None, height=None):
    """创建一个总是置顶的窗口"""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    
    if width and height:
        cv2.resizeWindow(window_name, width, height)

def main():
    args = parse_args()
    print(f"使用的 ONNX 提供者: {providers}")
    
    # 加载字体
    font_small, font_large = load_font()
    
    if args.image:
        # 图片模式
        img_path = args.image if os.path.isabs(args.image) else os.path.join(os.path.dirname(os.path.abspath(__file__)), args.image)
        print(f"尝试读取图片：{img_path}")
        frame = cv2.imread(img_path)
        
        if frame is None:
            print("图片读取失败，请检查路径是否正确！")
            return
            
        # 处理单帧
        result = process_frame(frame, app, font_small, font_large)
        
        # 创建窗口并显示结果
        window_name = "人脸分析结果 (按任意键退出)"
        create_window(window_name)
        cv2.imshow(window_name, result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 保存结果
        if args.output:
            cv2.imwrite(args.output, result)
            print(f"结果已保存为 {args.output}")
    else:
        # 摄像头模式
        print(f"尝试打开摄像头 {args.camera}")
        cap = cv2.VideoCapture(args.camera)
        
        if not cap.isOpened():
            print("无法打开摄像头！")
            return
            
        print("\n=== 操作说明 ===")
        print("1. 点击图像窗口使其获得焦点")
        print("2. 按 'q' 或 'ESC' 键退出程序")
        print("3. 按 's' 键保存当前帧")
        print("===============\n")
        
        # 创建置顶窗口
        window_name = "实时人脸分析 (q/ESC:退出 s:保存)"
        create_window(window_name)
        
        # 性能监控
        fps_counter = FPSCounter()
        frame_count = 0
        processing_scale = 0.5  # 处理时缩小到一半大小以提高性能
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法获取摄像头画面！")
                    break
                
                # 调整输入分辨率
                frame = cv2.resize(frame, (args.width, args.height))
                
                # 根据间隔决定是否处理这一帧
                if frame_count % args.interval == 0:
                    # 处理当前帧
                    result, process_time = process_frame(frame, app, font_small, font_large, processing_scale)
                    fps_counter.update(process_time)
                else:
                    result = frame
                    fps_counter.update()
                
                # 在画面上显示性能信息
                fps = fps_counter.get_fps()
                avg_process_time = fps_counter.get_processing_time()
                cv2.putText(result, f"FPS: {fps:.1f} Process: {avg_process_time:.1f}ms",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # 显示结果
                cv2.imshow(window_name, result)
                
                # 检查按键（支持 q 和 ESC）
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:  # 27 是 ESC 键的 ASCII 码
                    print("用户请求退出")
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    save_path = f"face_analysis_{timestamp}.jpg"
                    cv2.imwrite(save_path, result)
                    print(f"当前帧已保存为 {save_path}")
                
                frame_count += 1
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("已关闭摄像头")

if __name__ == "__main__":
    main()