# -*- coding:utf-8 -*-


import os
# server端
import socket
import sys

serverAddr = './uds_socket'  # 套接字存放路径及名称


def serverSocket():
    # create sockert
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)  # unix套接字，tcp通信方式
    if sock.fileno() < 0:
        print(sys.stderr, 'socket error')
    # bind to a file
    if os.path.exists(serverAddr):
        os.unlink(serverAddr)  # 如果套接字存在，则删除
    if sock.bind(serverAddr):  # 绑定套接字文件，绑定成功后，会在指定路径下生成一个域套接字文件。
        print(sys.stderr, 'socket.bind error')

    # listen
    if sock.listen(5):  # 最多监听5个客户端
        print(sys.stderr, 'socket.listen error')

    while True:
        print(sys.stderr, 'waiting for connecting')
        # waiting for client connecting
        conn, clientAddr = sock.accept()  # 如果监听到客户端连接，则调用accept接收这个连接并同时新建一个socket来和客户进行通信
        try:
            # receive data
            # send data to client
            while True:
                data = conn.recv(100)  # 接收100个字节长度的数据
                if data:
                    print(sys.stderr, '1--received "%s"' % data)

                    # print(data,type(data))
                    data2= bytes("0",  encoding='UTF-8')
                    print(data2,type(data2))
                    conn.sendall(data2)  # 发送数据



                    # conn.sendall(data)  # 发送数据
                else:
                    break
        except Exception as e:
            print(e)


if __name__ == "__main__":
    serverSocket()

