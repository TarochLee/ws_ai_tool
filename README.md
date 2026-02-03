# ai截图总结工具

## 一、编译方案

```
cmake -S . -B build -G Ninja
cmake --build build -j 1

# 运行（确保当前目录有 pic.jpg、models/GGUF/..）
./build/pic_brief

```