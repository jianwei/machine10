#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import uuid
from common.serial_control import serial_control
import sys
import time
ser = serial_control()


def main(cmd):
    cmd =cmd+"."
    cmd_dict = {
        "uuid": str(uuid.uuid1()),
        "cmd": cmd,
        "from": "camera",
    }
    print("cmd_dict",cmd_dict)
    ser.send_cmd(cmd_dict)
    print(ser.get_ret())



if __name__ == "__main__":
    try:
        
        cmd2 = "MF 100"
        print("------------------------------------------------------------------"+cmd2+"-------------------------------------------------------------------------")
        main(cmd2)
        time.sleep(1)
        cmd3 = "MF 15"
        print("------------------------------------------------------------------"+cmd3+"-------------------------------------------------------------------------")
        main(cmd3)

        cmd4 = "MF 100"
        print("------------------------------------------------------------------"+cmd4+"-------------------------------------------------------------------------")
        main(cmd4)
        # main("TL 90.")
        # main("STOP 0.")
        # main("STOP 0.")
    except KeyboardInterrupt:
        print("ctrl+c stop")
        # ser.close()

