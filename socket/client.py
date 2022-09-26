# -*- coding:utf-8 -*-

# Author: ChenTong
# Date: 2021/11/11 09:48


# client端
import socket
import sys

serverAddr = './uds_socket'  # 注意想要跟谁通信就绑定谁的套接字文件


def clientSocket():
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if sock.fileno() < 0:
        print(sys.stderr, 'socket error')

    try:
        sock.connect(serverAddr)
    except socket.error as msg:
        print(sys.stderr, "exception")
        print(sys.stderr, msg)
        sys.exit(1)

    message = b'this is the message'
    sock.sendall(message)

    amountRecv = 0
    amountSnd = len(message)

    # while amountRecv < amountSnd:
        # print()
    data = sock.recv(100)
    # amountRecv += len(data)
    print(sys.stderr, 'received "%s"' % data)
    sock.close()


if __name__ == "__main__":
    clientSocket()

