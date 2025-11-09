# Group05 Age-detection-face-recognition (local copy)

Usage / Documentation (English)

## Overview
This is a demo project for face age and gender detection based on Flask + InsightFace. You can upload images via the web UI or use your camera for real-time inference.

Project layout (short):
- `age/`: application source (e.g. `app.py`, templates)
- `age/uploads/`: processed result images (ignored by `.gitignore`)
- `README.md`: this file
- `使用文档.md`: original Chinese usage document

## Prerequisites
- Python 3.8+ (Windows / Linux / macOS)
- It's recommended to create a virtual environment (venv or conda) to isolate dependencies

## Install dependencies
A `requirements.txt` is included in the project root. To install dependencies:

```powershell
# after activating your virtual environment
pip install -r requirements.txt
```

Common dependencies include:
- flask
- flask-socketio
- eventlet (or gevent — choose the proper SocketIO server for production)
- insightface
- onnxruntime
- opencv-python
- pillow
- numpy

Note: InsightFace downloads model files the first time it runs (internet required). Installing onnxruntime with GPU support may require additional steps.

## Run the app (development)
1. Change to the application directory and run:

```powershell
cd D:\Code\age\age
python app.py
```

2. Open your browser and go to: http://127.0.0.1:5000

3. Upload an image or switch to camera mode for real-time frames.

## Common issues & fixes
- FileNotFoundError when accessing result files
   - Avoid running `app.py` from a different working directory. The application is configured to use an absolute `uploads` path relative to `app.py`. Run from the `age` directory:

```powershell
cd D:\Code\age\age
python app.py
```

- Camera not available
   - Ensure your camera drivers are installed and the browser has permission to use the camera (for the front-end camera mode).

- GitHub push errors like `Permission denied (publickey)` or HTTPS authentication failures
   - See the SSH or HTTPS + PAT instructions below to configure authentication before pushing.

## Optional: model files
Do not commit large model files to Git. Place local InsightFace models under `age/models/` and keep them ignored in `.gitignore`.

## Push to GitHub (quick tips)
1. SSH (recommended)
    - Generate an SSH key: `ssh-keygen -t ed25519 -C "your_email@example.com"`
    - Copy the public key (e.g. `%USERPROFILE%\\.ssh\\id_ed25519.pub`) to GitHub -> Settings -> SSH and GPG keys
    - Test connection: `ssh -T git@github.com`

2. HTTPS + Personal Access Token (PAT)
    - Create a Personal Access Token on GitHub with `repo` permissions
    - Set remote URL to HTTPS:
       `git remote set-url origin https://github.com/<user>/<repo>.git`
    - When pushing, provide your PAT as the password

## Notes
- The `uploads/` folder stores output images and is ignored by `.gitignore`.
- For production deployment use a proper WSGI server and static-file serving (e.g. nginx + gunicorn/uvicorn with a SocketIO adapter).

---
If you'd like, I can move the English documentation into a `docs/` folder, add a link in the README, or create a simple CI workflow (lint/tests) for the repository.

如何推送（SSH）：
1. 在本地生成 SSH key（如果还没有）：
   ssh-keygen -t ed25519 -C "your_email@example.com"
   # 然后把生成的公钥 id_ed25519.pub 内容复制到 GitHub Settings -> SSH and GPG keys
2. 测试连接：
   ssh -T git@github.com
3. 添加远程并推送：
   git remote add origin git@github.com:Lil3bear/Group05_Age-detection-face-recognition.git
   git branch -M main
   git push -u origin main

如何推送（HTTPS + PAT）：
1. 在 GitHub 生成一个 Personal Access Token（repo 权限）。
2. 使用 HTTPS remote：
   git remote add origin https://github.com/Lil3bear/Group05_Age-detection-face-recognition.git
3. 推送时会提示用户名和密码，把密码位置填入 PAT：
   git push -u origin main

如果我尝试推送并遇到认证失败，我会把错误信息贴出来并指导你如何解决。