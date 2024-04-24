import cv2
# import time
import os
import sys
import torch
import numpy as np

# 添加 YOLOv5 仓库的路径到 Python 的模块搜索路径中
sys.path.append('E:/BISHE shijuejiance/yolov5-5.0')

# 从 YOLOv5 的模型定义文件中导入所需的模型类
from models.experimental import attempt_load

# 加载预训练的模型权重
model = attempt_load('E:/BISHE shijuejiance/yolov5-5.0/weights/yolov5s.pt', map_location=torch.device('cpu'))

if not os.path.exists('img'):
    os.mkdir('img')

cap = cv2.VideoCapture(0)  # 调用摄像头'0'一般是打开电脑自带摄像头,'1'是打开外部摄像头(只有一个摄像头的情况)

if not cap.isOpened():
    print("无法打开摄像头")
    sys.exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 5472)  # 设置图像宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3648)  # 设置图像高度
cap.set(cv2.CAP_PROP_FPS, 15)
cap.set(cv2.CAP_PROP_ZOOM, 50)  # 调整放大倍数
cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # 调整曝光度


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = 'E:/BISHE shijuejiance/yolov5-5.0/weights/best.pt'
model = torch.load(model_path, map_location=device)['model'].float().fuse().eval()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 调整帧的形状以匹配模型的期望
    frame = frame.transpose((2, 0, 1))  # 将通道维度移到第一个位置
    frame = frame[None, ...]  # 添加批次维度

    # 将帧的数据类型转换为浮点数
    frame = frame.astype(np.float32)

    # 将帧从NumPy数组转换为PyTorch张量
    frame_tensor = torch.from_numpy(frame).to(device)

    # 将张量传递给模型
    results = model(frame_tensor)


    # 使用YOLOv5进行目标检测
    results = model(frame)

    # 在帧上绘制检测结果
    for *xyxy, conf, cls in results.xyxy[0]:
        label = f'{model.model.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
        cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # 显示帧
    cv2.imshow('YOLOv5 Real-time Object Detection', frame)

    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源并关闭窗口
cap.release()
cv2.destroyAllWindows()
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("无法读取视频帧")
#         break
#
#     frame_resized = cv2.resize(frame, (640, 480))
#     cv2.imshow("frame", frame_resized)
#
#     # 将捕获的帧转换为 RGB 颜色空间,确保数据是连续的,并将其转换为 PyTorch 张量
#     frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
#     frame_rgb = np.ascontiguousarray(frame_rgb)
#     frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0  # 归一化
#     frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # 调整维度
#
#     # 使用模型进行推理
#     with torch.no_grad():
#         results = model(frame_tensor)
#
#     # 处理检测结果
#     for detection in results:
#         if len(detection) == 6:  # 确保每个检测结果包含6个元素
#             x1, y1, x2, y2, conf, cls = detection
#             if conf > 0.3:  # 置信度阈值
#                 cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#
#
#
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord('c'):  # 按 'c' 键拍照并保存
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
#         filename = f"img/capture_{timestamp}.jpg"
#         cv2.imwrite(filename, frame)
#         print(f"图像已保存: {filename}")
#
# cap.release()  # 释放摄像头
# cv2.destroyAllWindows()  # 销毁窗口