# -*- coding:utf-8 -*-
import os
import socket
import sys
import time,uuid
from common.serial_control import serial_control

# serverAddr = '../uds_socket'  # 套接字存放路径及名称

class unix_socket():
    def __init__(self):
        self.ser = serial_control()
        self.server_addr = '../uds_socket'  # 套接字存放路径及名称
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)  # unix套接字，tcp通信方式
        

    def server(self):
        sock = self.socket
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
            print("--------------------------------------------begin-------------------------------------------------------")
            print(sys.stderr, 'waiting for connecting')
            conn, clientAddr = sock.accept()  
            try:
                while True:
                    message = conn.recv(100)
                    # print("time11:",time.time())
                    if message:
                        message = str(message, 'UTF-8')
                        print(sys.stderr, 'received "%s"' % message)
                        cmd_dict = {
                            "uuid": str(uuid.uuid1()),
                            "cmd": message,
                            "from": "camera",
                        }

                        ret  = self.ser.send_cmd(cmd_dict)
                        # ret  = "0"
                        # ret = bytes(ret,encoding='UTF-8')
                        # conn.sendall(ret)
                        print("--------------------------------------------end-------------------------------------------------------")
                    else:
                        break
            except Exception as e:
                print(e)

            pass
    
    def send_message(self,message):
        sock = self.socket
        if sock.fileno() < 0:
            print(sys.stderr, 'socket error')
        try:
            sock.connect(self.server_addr)
        except socket.error as msg:
            print(sys.stderr, "exception")
            print(sys.stderr, msg)
            sys.exit(1)

        reasult = ""
        print("send_message:",message)
        if(message):
            message = bytes(message,encoding='UTF-8')
            sock.sendall(message)
            # reasult = sock.recv(1)
            # print(sys.stderr, 'received "%s"' % reasult)
        sock.close()
        return reasult
           
        

