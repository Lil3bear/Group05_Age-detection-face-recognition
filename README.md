# Group05 Age-detection-face-recognition (local copy)

这是你的本地项目副本，包含一个基于 Flask + InsightFace 的人脸年龄/性别检测示例。

快速链接：

- 使用文档（中文）： `使用文档.md`

将此仓库推送到 GitHub 远程仓库前请注意：

- 请确保不上传大型模型文件（如 .onnx）或虚拟环境目录（已在 .gitignore 中列出）。
- 推送到 GitHub 需要 SSH key 或 HTTPS + PAT 认证；如果你没有配置，请参考下面的“如何推送”部分。

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