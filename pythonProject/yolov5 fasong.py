import torch
import cv2
import serial

# 加载YOLOv5模型
model = torch.hub.load('E:/BISHE shijuejiance/yolov5-5.0/yolov5', 'custom', path='best.pt')

# 打开视频流
cap = cv2.VideoCapture(0)


def send_coordinates(x, y):
    # 创建串口连接
    ser = serial.Serial('COM3', 9600)  # 替换为实际的串口号和波特率

    # 发送坐标数据
    message = f"{x},{y}\n"
    ser.write(message.encode())

    # 关闭连接
    ser.close()


while True:
    # 读取一帧图像
    ret, frame = cap.read()

    # 对图像进行推理
    results = model(frame)

    # 解析检测结果
    for detection in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = detection
        # 计算物体中心坐标
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        # 发送坐标给机器人控制器(使用USB串口通信)
        send_coordinates(x_center, y_center)

    # 显示检测结果
    cv2.imshow('YOLOv5', results.render()[0])

    if cv2.waitKey(1) == ord('q'):
        break
