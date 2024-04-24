import cv2
from yolov5 import YOLOv5

model = YOLOv5("E:/BISHE shijuejiance/yolov5-5.0/weights/yolov5s.pt", device='cpu')

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 5472)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3648)
# cap.set(cv2.CAP_PROP_FPS, 15)
# cap.set(cv2.CAP_PROP_ZOOM, 5)  # 调整放大倍数
# cap.set(cv2.CAP_PROP_EXPOSURE, -10)  # 调整曝光度

# coding=utf-8
import cv2
import time
import os

if os.path.exists('img') == False:
    os.mkdir('img')
filenames = os.listdir(r'img')

if __name__ == '__main__':

    # 开启ip摄像头
    cv2.namedWindow("camera", 1)
    # 这个地址就是下面记下来的局域网IP
    video = "http://admin:admin@192.168.31.92:8081/"  # 此处@后的ipv4 地址需要修改为自己的地址

    capture = cv2.VideoCapture(video)

    num = len(filenames)
    index = 0
    imgname = -1
    while True:
        success, img = capture.read()

        # 不进行旋转
        cv2.imshow("camera", img)

        # 获取长宽
        # (h, w) = img.shape[:2]
        # center = (w // 2, h // 2)
        # 进行旋转
        # M = cv2.getRotationMatrix2D(center, -90, 1.0)
        # rotated = cv2.warpAffine(img, M, (w, h))
        # 若不关参数，参数也会被旋转，影响效果
        # cv2.imshow("camera", rotated)

        # 按键处理，注意，焦点应当在摄像头窗口，不是在终端命令行窗口
        key = cv2.waitKey(10)

        if key == 27:
            # 按esc键退出
            print("esc break...")
            break

        if key == ord(' '):
            # 按空格 保存图像 图片的路径
            while True:
                index = index + 1
                success, img = capture.read()
                cv2.imshow("camera", img)
                cv2.waitKey(10)
                if index == 15:
                    num = num + 1
                    imgname = imgname + 1
                    if imgname == -1:  # 此处改为-1为无限截取图片
                        break
                    filename = "img\\frames_%s.jpg" % (num)
                    cv2.imwrite(filename, img)
                    index = 0

    capture.release()
    cv2.destroyWindow("camera")

# 显示图像
# while True:
#     ret, frame = cap.read()
#     # print(ret)  #
#     ########图像不处理的情况
#     frame_1 = cv2.resize(frame, (640, 512))
#     cv2.imshow("frame", frame_1)
#
#     input = cv2.waitKey(1)
#     # if input == ord('q'):
#     #     break

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 缩放帧到模型输入尺寸
    input_frame = cv2.resize(frame, (640, 512))

    # 使用YOLOv5进行目标检测
    results = model.predict(input_frame)

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

cap.release()
cv2.destroyAllWindows()