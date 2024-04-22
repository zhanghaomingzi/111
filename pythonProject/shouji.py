# coding=utf-8
import cv2
import time
import os
import sys
import torch

# 添加 YOLOv5 仓库的路径到 Python 的模块搜索路径中
sys.path.append('E:/BISHE shijuejiance/yolov5-5.0')

# 从 YOLOv5 的模型定义文件中导入所需的模型类
from models.yolo import Model

# 加载预训练的模型权重
model = torch.load('E:/BISHE shijuejiance/yolov5-5.0/weights/yolov5s.pt')

if os.path.exists('img') == False:
    os.mkdir('img')
import cv2
import torch
from shexiangtoudiaoyong import access_get_image

# 加载YOLOv5模型
model = torch.load('E:/BISHE shijuejiance/yolov5-5.0/weights/yolov5s.pt')

while True:
    # 从sj.py中获取视频帧
    frame = access_get_image()

    if frame is None:
        break

    # 对帧进行YOLOv5检测
    results = model(frame)

    # 处理检测结果
    detections = results.xyxy[0].cpu().numpy()
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        label = model.names[int(cls)]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示处理后的帧
    cv2.imshow("YOLOv5 Detection", frame)

    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cv2.destroyAllWindows()
