import cv2
from yolov5 import YOLOv5
# import sys
#
#
# # 添加 YOLOv5 仓库的路径到 Python 的模块搜索路径中
# sys.path.append('E:/BISHE shijuejiance/yolov5-5.0')
# 加载预训练的YOLOv5模型
model = YOLOv5("E:/BISHE shijuejiance/yolov5-5.0/weights/yolov5s.pt",device='cpu')  # 选择模型

# 打开摄像头
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 5472)  # 设置图像宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3648)  # 设置图像高度
cap.set(cv2.CAP_PROP_FPS, 30)  # 设置帧率


while True:
    # 从摄像头读取帧
    ret, frame = cap.read()

    if not ret:
        break

    # 使用YOLOv5进行目标检测
    results = model.predict(frame)

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
