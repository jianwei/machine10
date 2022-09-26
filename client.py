from common.unix_socket import unix_socket
import time
s = unix_socket()
message = "MF 40."
print("time40-1:",time.time())
s.send_message(message)

time.sleep(2)
s = unix_socket()
print("time50-1:",time.time())
message = "MF 50."
s.send_message(message)