from common.unix_socket import unix_socket

s = unix_socket()
message = "MF 40."
s.send_message(message)