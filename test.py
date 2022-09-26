#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import uuid
# from common.serial_control import serial_control
import sys
import time
# ser = serial_control()


def main(cmd):
    cmd =cmd+"."
    cmd_dict = {
        "uuid": str(uuid.uuid1()),
        "cmd": cmd,
        "from": "camera",
    }
    print("cmd_dict",cmd_dict)
    # ser.send_cmd(cmd_dict)
    # print(ser.get_ret())





def turn(data):
    last = data[0]
    total = 0
    for i in range(len(last)):
        item  = last[i]
        total += float(item.get("centerx"))
    print("total:{},average:{}".format(total,total/len(last)))
    # print(total)





if __name__ == "__main__":
    try:
        data = [
                    [
                        {"point": [[336, 341], [387, 341], [336, 384], [387, 384]], "camera_id": 1, "track_id": 0, "name": "box", "time": 1664162673.668983, "screenSize": [640, 480], "centerx": 362.5, "centery": 362.5, "center": [361.5, 362.5]},
                        {"point": [[336, 341], [387, 341], [336, 384], [387, 384]], "camera_id": 1, "track_id": 0, "name": "box", "time": 1664162673.668983, "screenSize": [640, 480], "centerx": 358.5, "centery": 362.5, "center": [361.5, 362.5]},
                        {"point": [[336, 341], [387, 341], [336, 384], [387, 384]], "camera_id": 1, "track_id": 0, "name": "box", "time": 1664162673.668983, "screenSize": [640, 480], "centerx": 342.5, "centery": 362.5, "center": [361.5, 362.5]},
                        {"point": [[336, 341], [387, 341], [336, 384], [387, 384]], "camera_id": 1, "track_id": 0, "name": "box", "time": 1664162673.668983, "screenSize": [640, 480], "centerx": 367.5, "centery": 362.5, "center": [361.5, 362.5]},
                        {"point": [[336, 341], [387, 341], [336, 384], [387, 384]], "camera_id": 1, "track_id": 0, "name": "box", "time": 1664162673.668983, "screenSize": [640, 480], "centerx": 388.5, "centery": 362.5, "center": [361.5, 362.5]},
                        {"point": [[336, 341], [387, 341], [336, 384], [387, 384]], "camera_id": 1, "track_id": 0, "name": "box", "time": 1664162673.668983, "screenSize": [640, 480], "centerx": 368.5, "centery": 362.5, "center": [361.5, 362.5]},
                    ],    
                ]
        turn (data)
        # cmd2 = "MF 30"
        # print("------------------------------------------------------------------"+cmd2+"-------------------------------------------------------------------------")
        # main(cmd2)
        # time.sleep(1)

        # cmd2 = "MF 20"
        # print("------------------------------------------------------------------"+cmd2+"-------------------------------------------------------------------------")
        # main(cmd2)
        # time.sleep(1)

      
        

        # main("TL 90.")
        # main("STOP 0")
        # main("STOP 0.")
    except KeyboardInterrupt:
        print("ctrl+c stop")
        # ser.close()

