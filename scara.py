# import socket
#
# # 机器人的IP地址和端口号
# robot_ip = '192.168.0.2'
# robot_port = 201
#
# # 建立TCP/IP连接
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.connect((robot_ip, robot_port))
#
# # 发送控制命令
# command = 'MOVJ 0 20 0 0'  # 示例命令:移动到关节角度(0, 90, 0, 0)
# sock.send(command.encode())
#
# # 接收机器人的响应
# response = sock.recv(1024).decode()
# print('Robot response:', response)
#
# # 关闭连接
# sock.close()
import serial

# 设置串口参数
port = 'COM1'
baudrate = 9600
bytesize = serial.EIGHTBITS
parity = serial.PARITY_NONE
stopbits = serial.STOPBITS_ONE

# 建立串口连接
ser = serial.Serial(port=port, baudrate=baudrate, bytesize=bytesize, parity=parity, stopbits=stopbits)

# 发送命令
command = 'MOVE X100 Y200 Z300\r\n'
ser.write(command.encode('ascii'))

# 接收响应
response = ser.readline().decode('ascii').strip()
print('Received:', response)

# 关闭串口连接
ser.close()