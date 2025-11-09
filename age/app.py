import os
import cv2
import time
import uuid
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
from insightface.app import FaceAnalysis
import base64
import io

app = Flask(__name__)
# 使用绝对路径
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB
socketio = SocketIO(app, cors_allowed_origins="*")

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

print(f"上传文件夹路径: {app.config['UPLOAD_FOLDER']}")

# 初始化人脸分析模型
providers = ['CPUExecutionProvider']  # 可以根据需要改为 CUDA
face_analyzer = FaceAnalysis(providers=providers)
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# 加载字体
def get_fonts():
    try:
        font_path = "C:/Windows/Fonts/sarasa-mono-sc-regular.ttf"
        if not os.path.exists(font_path):
            font_path = "C:/Windows/Fonts/msyh.ttc"
        
        if not os.path.exists(font_path):
            raise FileNotFoundError("找不到中文字体文件")
            
        return ImageFont.truetype(font_path, 40), ImageFont.truetype(font_path, 60)
    except Exception as e:
        print(f"加载字体出错: {str(e)}")
        return None, None

font_small, font_large = get_fonts()

def process_image(image_path):
    """处理图片并返回结果"""
    try:
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            return None, "无法读取图片"

        # 调整大小以提高性能
        max_size = 1024
        height, width = img.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            img = cv2.resize(img, None, fx=scale, fy=scale)
            
        # 确保图片不是空的
        if img.size == 0:
            return None, "图片内容为空"
            
        # 分析人脸
        faces = face_analyzer.get(img)
        if not faces:
            return None, "未检测到人脸"
            
        # 处理检测到的人脸
        for face in faces:
            bbox = face.bbox.astype(int)
            age = int(face.age)
            gender = "男" if face.gender == 1 else "女"

            # 绘制人脸框
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # 转换为PIL图像以支持中文
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            # 计算文字位置
            y_text = max(bbox[1] - 70, 10)  # 确保不会超出图片上边界
            
            if font_small and font_large:
                # 绘制性别
                gender_text = f"{gender},"
                gender_bbox = draw.textbbox((0, 0), gender_text, font=font_small)
                gender_width = gender_bbox[2] - gender_bbox[0]
                draw.text((bbox[0], y_text), gender_text, fill=(255, 0, 0), font=font_small)

                # 绘制年龄
                age_text = f"{age}岁"
                draw.text((bbox[0] + gender_width + 10, y_text - 10), age_text, fill=(255, 0, 0), font=font_large)

            # 转回OpenCV格式
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 保存结果
        result_filename = f"result_{str(uuid.uuid4())[:8]}.jpg"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        
        # 确保文件成功写入
        if cv2.imwrite(result_path, img):
            # 等待文件写入完成
            time.sleep(0.1)
            if os.path.exists(result_path):
                return result_filename, None
            else:
                return None, "保存结果图片失败：文件未能成功写入"
        else:
            return None, "保存结果图片失败"
            
    except Exception as e:
        print(f"处理图片时出错: {str(e)}")
        return None, f"处理图片时出错: {str(e)}"
    
    # 处理每个检测到的人脸
    for face in faces:
        bbox = face.bbox.astype(int)
        age = int(face.age)
        gender = "男" if face.gender == 1 else "女"

        # 绘制人脸框
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # 转换为PIL图像以支持中文
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 计算文字位置
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
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 保存结果
    result_filename = f"result_{str(uuid.uuid4())[:8]}.jpg"
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_path, img)
    
    return result_filename, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件被上传'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'})
    
    if file:
        try:
            # 安全地保存文件
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(temp_path)
            
            # 处理图片
            result_filename, error = process_image(temp_path)
            
            # 删除临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            if error or not result_filename:
                return jsonify({'error': error or '处理图片失败'})
            
            # 确保结果文件存在
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            if not os.path.exists(result_path):
                return jsonify({'error': '生成的图片文件未能找到'})
            
            return jsonify({
                'success': True,
                'result_url': f'/uploads/{result_filename}'
            })
        except Exception as e:
            # 确保清理临时文件
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            return jsonify({'error': f'处理图片时出错: {str(e)}'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': '找不到请求的文件'}), 404
    return send_file(file_path)

def process_frame_realtime(frame_data):
    """处理实时视频帧"""
    try:
        # 解码Base64图像
        encoded_data = frame_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 调整大小以提高性能
        height, width = img.shape[:2]
        max_size = 640
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            img = cv2.resize(img, None, fx=scale, fy=scale)

        # 分析人脸
        faces = face_analyzer.get(img)
        
        # 处理每个检测到的人脸
        for face in faces:
            bbox = face.bbox.astype(int)
            age = int(face.age)
            gender = "男" if face.gender == 1 else "女"

            # 绘制人脸框
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # 转换为PIL图像以支持中文
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            # 计算文字位置
            y_text = bbox[1] - 70
            if y_text < 20:
                y_text = bbox[3] + 30

            # 绘制性别和年龄
            gender_text = f"{gender},"
            gender_bbox = draw.textbbox((0, 0), gender_text, font=font_small)
            gender_width = gender_bbox[2] - gender_bbox[0]
            draw.text((bbox[0], y_text), gender_text, fill=(255, 0, 0), font=font_small)
            
            age_text = f"{age}岁"
            draw.text((bbox[0] + gender_width + 10, y_text - 10), age_text, fill=(255, 0, 0), font=font_large)

            # 转回OpenCV格式
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 转换为JPEG格式的Base64
        _, buffer = cv2.imencode('.jpg', img)
        jpg_as_text = base64.b64encode(buffer).decode()
        return 'data:image/jpeg;base64,' + jpg_as_text

    except Exception as e:
        print(f"处理帧时出错: {str(e)}")
        return None

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('frame')
def handle_frame(frame_data):
    """处理从客户端接收的视频帧"""
    result = process_frame_realtime(frame_data)
    if result:
        emit('processed_frame', result)

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)