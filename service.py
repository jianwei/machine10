from common.unix_socket import unix_socket

try:
    s = unix_socket()
    s.server()
except KeyboardInterrupt:
    print("ctrl+c stop")
    # s.send_message("STOP 0")
