# ws_ai_tool

本项目是一个本地运行的“图片 → OCR → LLM 总结/知识扩展”的轻量服务。启动后会监听本地端口（默认 `8080`），通过浏览器页面上传图片、粘贴剪贴板图片或拖拽图片，后端异步执行 OCR 与大模型推理，并在前端展示进度条与最终输出文本。

## 功能特性

- 本地 HTTP 服务：默认 `http://127.0.0.1:8080`
- Web 交互界面：
  - 上传图片文件（PNG/JPG/WebP/HEIC 等）
  - Ctrl+V 粘贴剪贴板图片
  - 拖拽图片到页面
- OCR：使用 macOS Vision 框架（中英文识别）
- LLM：基于 `llama.cpp`（支持 Metal 加速）
- 异步任务队列：
  - 提交任务后立刻返回 `task_id`
  - 前端轮询 `/api/status` 展示进度条、状态与结果
- 输出格式：固定两段中文
  - 第一段：对截图文字总结（不分点）
  - 第二段：相关扩展知识（不分点）

## 项目结构（建议）


## 构建依赖

- macOS（推荐 Apple Silicon：M1/M2/M3）
- Xcode Command Line Tools
- CMake ≥ 3.20
- C++17
- macOS Frameworks：
  - Foundation
  - Vision
  - CoreGraphics
  - ImageIO
- `llama.cpp`（项目内子模块或外部依赖）
- `cpp-httplib`（本项目使用单头文件 `httplib.h`）

## 编译

以 `b/` 为构建目录示例：

```
cmake -S . -B b     
cmake --build b -j 1

# 运行（确保当前目录有 pic.jpg、models/GGUF/..）
./b/src/ws_ai_server

```

浏览器访问：

```
http://127.0.0.1:8080/
```

## License

暂无