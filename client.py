from common.unix_socket import unix_socket

s = unix_socket()
message = b'this is the message--1'
s.send_message(message)