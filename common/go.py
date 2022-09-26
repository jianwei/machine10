from common.serial_control import serial_control
import json
import time
import uuid
from common.unix_socket import unix_socket

class go ():
    def __init__(self, redis):
        # self.ser = serial_control()
        self.unix_socket = unix_socket()
        self.redis = redis
        self.default_machine_speed = 5
        self.current_machine_speed = 5
        self.increment = 1     # 速度增量
        self.min_unit_px = 5  # 每秒行驶多少像素
        self.last_check_time = float(time.time())
        self.last_turn_time = float(time.time())

        # self.server_address = './socket/uds_socket'

    def send_comand(self, cmd):
        ret = ""
        if (cmd != ""):
            cmd += "."
            print("cmd {}".format(cmd))
            self.unix_socket.send_message(cmd)
            # cmd_dict = {
            #     "uuid": str(uuid.uuid1()),
            #     "cmd": cmd,
            #     "from": "camera",
            # }
            # print(cmd)
            # self.ser = serial_control()
            # ret = self.ser.send_cmd(cmd_dict)
            # self.ser.close()
        else:
            print("cmd null")
        return ret
    
    def stop(self):
        self.send_comand("STOP 0")
    
    def set_default_speed(self):
        self.send_comand("MF "+str(self.default_machine_speed))

    def turn(self):
        pass

    def is_add_speed(self, redis_key):
        # print("redis_key",redis_key)
        data = self.redis.get(redis_key)
        if (data and len(data)>=2):
            data = json.loads(data)
            first = data[0]
            second = data[1]
            index_list = [-1, -1]
            flag = False
            for i in range(len(first)):
                index = first[i]["track_id"]
                for j in range(len(second)):
                    index2 = second[j]["track_id"]
                    if (index == index2):
                        flag = True
                        index_list[1] = j
                        break
                if (flag):
                    index_list[0] = i
                    break
            if (not index_list[0] == -1) and (not index_list[1] == -1):
                first_data = first[index_list[0]]
                second_data = second[index_list[1]]
                time1= first_data.get("time")
                time2= second_data.get("time")
                difftime  = time2-time1
                diff_px = abs(second_data.get("centery")-first_data.get("centery"))
                diff_time = abs(second_data.get("time")-first_data.get("time"))
                now = float(time.time())
                last_diff_time  = now - self.last_check_time
                print("-------------------------------------------------------------------------------------------------------")
                print("last_diff_time:{},difftime:{},diff_px:{},diff_px/diff_time:{}".format(last_diff_time,difftime,diff_px,diff_px/diff_time))
                print("-------------------------------------------------------------------------------------------------------")
                if (float(time.time()) - self.last_check_time >1) :
                    self.last_check_time  = now 
                    if (abs(diff_px/diff_time) < self.min_unit_px):
                        self.current_machine_speed += self.increment
                        self.current_machine_speed = self.current_machine_speed if  self.current_machine_speed <=25 else 25
                        self.send_comand("MF "+str(self.current_machine_speed))
                    else:
                        self.current_machine_speed = self.default_machine_speed
                        self.send_comand("MF "+str(self.default_machine_speed))
                else:
                    print("1秒内，不做处理")
