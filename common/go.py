from common.serial_control import serial_control
import json
import time
import uuid
from common.unix_socket import unix_socket

class go ():
    def __init__(self, redis):
        # self.ser = serial_control()
        
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
            self.unix_socket = unix_socket()
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
    
    # def 


    def turn(self,redis_key):
        data = self.redis.get(redis_key)
        if (data and len(data)>=2):
            data = json.loads(data)
            first = data[0]
            total = 0
            for item in first:
                total += float(item.get("centerx"))
            avg_centerx = total/len(first)

            
            unit = 0.0386  # 1 pint 0.0386cm
            gap = 30  # cm 导航摄像头的视野盲区
            if (point):
                cmd = ""
                centerx = point["centerx"]
                centery = point["centery"]
                screenSize = point["screenSize"]
                center_point = screenSize[0]/2
                diff_point_x = centerx-center_point
                tan = (diff_point_x*unit)/(gap+centery*unit)
                angle = int(numpy.arctan(tan) * 180.0 / 3.1415926)
                global_angle = self.global_angle
                print("global_angle:",global_angle)
                cmd_prefix = ""
                target_angle = 90
                if (global_angle <= 90):
                    if (centerx <= center_point):
                        target_angle = 90-angle
                        cmd_prefix = "TR" if global_angle < target_angle else "TL"
                    else:
                        target_angle = 90+angle
                        cmd_prefix = "TR"
                else:
                    if (centerx <= center_point):
                        target_angle = 90-angle
                        cmd_prefix = "TL"
                    else:
                        target_angle = 90+angle
                        cmd_prefix = "TR" if global_angle < target_angle else "TL"
                # print("target_angle,global_angle5",target_angle,global_angle)
                if (target_angle != global_angle):
                    cmd = cmd_prefix + " " + str(abs(target_angle-global_angle))
                    global_angle = target_angle
                    print("send-cmd:", cmd)
                else:
                    print("send-cmd:none")
                # print("cmd:",cmd)
                turn_ret = self.send(cmd)
                print("cmd,turn_ret:", cmd, turn_ret)
                # if(turn_ret==0 or turn_ret=="0") :
                # 继续前行
                self.send(cmd)
    



                
