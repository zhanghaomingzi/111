import cv2
from shexiangtoudiaoyong import access_get_image
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
import numpy as np

model = attempt_load('E:/BISHE shijuejiance/yolov5-5.0/weights/yolov5s.pt', map_location='cpu')

cam = cv2.VideoCapture(0)  # 使用默认摄像头 (索引为0)

while True:
    frame = access_get_image(cam)
    if frame is not None:
        # 对获取到的帧进行预处理
        img = cv2.resize(frame, (640, 640))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float() / 255.0  # 归一化
        img = img.unsqueeze(0)

        # 对预处理后的帧进行目标检测
        results = model(img)

        # 对检测结果进行后处理
        results = non_max_suppression(results, conf_thres=0.5, iou_thres=0.5)

        # 绘制检测结果
        for det in results:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, frame, label=label, color=(0, 255, 0), line_thickness=2)

        cv2.imshow('Frame', frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("无法获取摄像头帧")
        break

# 释放摄像头并关闭窗口
cam.release()
cv2.destroyAllWindows()