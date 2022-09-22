from common.serial_control import serial_control
import json
import time
import uuid


class go ():
    def __init__(self, redis):
        self.ser = serial_control()
        self.redis = redis
        self.default_machine_speed = 15
        self.current_machine_speed = 15
        self.increment = 10        # 速度增量
        pass

    def send_comand(self, cmd):
        ret = ""
        if (cmd != ""):
            cmd += "."
            cmd_dict = {
                "uuid": str(uuid.uuid1()),
                "cmd": cmd,
                "from": "camera",
            }
            print(cmd)
            # self.ser = serial_control()
            ret = self.ser.send_cmd(cmd_dict)
            # self.ser.close()
        else:
            print("cmd null")
        return ret

    def is_add_speed(self, redis_key):
        print("redis_key",redis_key)
        data = self.redis.get(redis_key)
        if (data):
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
                # print(first_data.get("centery"),second_data.get("centery"),second_data.get("centery")-first_data.get("centery"))
                if (second_data.get("centery")-first_data.get("centery") < 10):
                    self.current_machine_speed += self.increment
                    self.send_comand("MF "+str(self.current_machine_speed))
                else:
                    self.current_machine_speed = self.default_machine_speed
                    self.send_comand("MF "+str(self.default_machine_speed))
                pass
