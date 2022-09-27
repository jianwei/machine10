import json
import time
import numpy
from common.unix_socket import unix_socket
import functools


def cmpy(a, b):
    return b.get("centery")-a.get("centery")

class go ():
    def __init__(self, redis):
        self.redis = redis
        self.default_machine_speed = 5
        self.current_machine_speed = 5
        self.increment = 1     # 速度增量
        self.min_unit_px = 5  # 每秒行驶多少像素
        self.last_check_time = float(time.time())
        self.last_turn_time = float(time.time())
        self.global_angle = 90
        # self.server_address = './socket/uds_socket'
    

    

    def send_comand(self, cmd):
        ret = ""
        if (cmd != ""):
            cmd += "."
            print("cmd {}".format(cmd))
            self.unix_socket = unix_socket()
            self.unix_socket.send_message(cmd)
        else:
            print("cmd null")
        return ret
    
    def stop(self):
        self.send_comand("STOP 0")
    
    def set_default_speed(self):
        print("go.py set_default_speed")
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
                    print("1秒内，不重复加速")
    
    def get_target_x (self,frame):
        total = 0
        for item in frame:
            total += float(item.get("centerx"))
        avg_centerx = total/len(frame)
        return avg_centerx


    def turn(self,redis_key):
        now = float(time.time())
        if(now-self.last_turn_time<1):
            print("1秒内不重复转向:",self.last_turn_time,now)
            self.last_turn_time = now
            return

        data = self.redis.get(redis_key)
        if (data and len(data)>=2):
            data = json.loads(data)
            first = data[0]
            first.sort(key=functools.cmp_to_key(cmpy))
            # print("first:",first)

            avg_centerx = self.get_target_x(first)
            # 15*11
            # [[314, 427], [375, 427], [314, 479], [375, 479]]
            unit = 0.2115  # 1 pint 0.2115cm
            gap = 115  # cm 导航摄像头的视野盲区
            cmd = ""
            point = first[0]
            centerx = avg_centerx
            
            centery = point["centery"]
            screenSize = point["screenSize"]
            # center_point = avg_centerx
            center_point = screenSize[0]/2
            diff_point_x = centerx-center_point
            diff_point_y = point["screenSize"][1]-centery
            # print("diff_point_x*unit:{},gap+centery*unit:{},centery:{},diff_point_x:{},screenSize:{},diff_point_y:{}".format(diff_point_x*unit,gap+diff_point_y*unit,centery,diff_point_x,point["screenSize"],diff_point_y))
            tan = (diff_point_x*unit)/(gap+diff_point_y*unit)
            angle = int(numpy.arctan(tan) * 180.0 / 3.1415926)
            global_angle = self.global_angle
            # print("global_angle:",global_angle)
            cmd_prefix = ""
            target_angle = 90
            if (global_angle <= 90):
                if (centerx <= center_point):
                    target_angle = 90-angle
                    cmd_prefix = "TL" if global_angle < target_angle else "TR"
                else:
                    target_angle = 90+angle
                    cmd_prefix = "TL"
            else:
                if (centerx <= center_point):
                    target_angle = 90-angle
                    cmd_prefix = "TL"
                else:
                    target_angle = 90+angle
                    cmd_prefix = "TL" if global_angle < target_angle else "TR"
            # print("target_angle,global_angle5",target_angle,global_angle)
            # print("target_angle:{},global_angle:{}".format(target_angle,global_angle))
            if (target_angle != global_angle):
                cmd = cmd_prefix + " " + str(abs(target_angle-global_angle))
                self.global_angle = target_angle
                print("send-cmd:", cmd)
                self.send_comand(cmd)
            else:
                print("send-cmd:none")