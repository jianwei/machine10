# -*- coding:utf-8 -*-
import os
import socket
import sys
from common.serial_control import serial_control

# serverAddr = '../uds_socket'  # 套接字存放路径及名称

class unix_socket():
    def __init__(self):
        self.ser = serial_control()
        self.server_addr = '../uds_socket'  # 套接字存放路径及名称
        

    def server(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)  # unix套接字，tcp通信方式
        if sock.fileno() < 0:
            print(sys.stderr, 'socket error')
        if os.path.exists(self.server_addr):
            os.unlink(self.server_addr)  # 如果套接字存在，则删除
        if sock.bind(self.server_addr):  # 绑定套接字文件，绑定成功后，会在指定路径下生成一个域套接字文件。
            print(sys.stderr, 'socket.bind error')

        # listen
        if sock.listen(5):  # 最多监听5个客户端
            print(sys.stderr, 'socket.listen error')

        while True:
            print(sys.stderr, 'waiting for connecting')
            conn, clientAddr = sock.accept()  
            try:
                while True:
                    message = conn.recv(100)  # 接收100个字节长度的数据
                    if message:
                        print(sys.stderr, 'received "%s"' % message)
                        ret  = self.ser.send_cmd(message)
                        ret = bytes(ret,encoding='UTF-8')
                        conn.sendall(ret)  # 发送数据
                    else:
                        break
            except Exception as e:
                print(e)

            pass
    
    def send_message(self,message):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if sock.fileno() < 0:
            print(sys.stderr, 'socket error')

        try:
            sock.connect(self.server_addr)
        except socket.error as msg:
            print(sys.stderr, "exception")
            print(sys.stderr, msg)
            sys.exit(1)

        reasult = ""
        if(message):
            # ret  = self.ser.send_cmd(message)
            message = bytes(message,encoding='UTF-8')
            sock.sendall(message)
            reasult = sock.recv(100)
        return reasult
            # print(sys.stderr, 'received "%s"' % data)
        

