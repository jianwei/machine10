# coding:utf-8
import json
import time
from common.unix_socket import unix_socket
import functools
import uuid

def cmpy(a, b):
    return b.get("centery")-a.get("centery")

class work_space():
    def __init__(self, redis): 
        self.redis = redis
        self.uuid = str(uuid.uuid1())
        

    def send_cmd(self, cmd):
        ret = ""
        if (cmd != ""):
            cmd += "."
            print("work.py,cmd {}".format(cmd))
            self.unix_socket = unix_socket()
            self.unix_socket.send_message(cmd)
        else:
            print("work.py,cmd null")
        return ret


    

    def wheel(self, redis_key,speed):
        vegetable_points = self.redis.get(redis_key)
        if (vegetable_points):
            vegetable_points = json.loads(vegetable_points)
            first_frame = vegetable_points[0]
            first_frame.sort(key=functools.cmp_to_key(cmpy))
            first = first_frame[0]
            centery = first["centery"]
            track_id = first["track_id"]
            key = "done_"+self.uuid+"_"+str(track_id)
            has_done = self.redis.get(key)
            print("work.py,key:{},has_done:{}".format(key,has_done))
            if (has_done==""):
                if centery >= 50 and centery <= 250:
                    rot_speed = 60
                    unit_sleep = 1 / (rot_speed * 50 / 2 / 1000)  # 转1圈所需要的时间
                    # unit_sleep -= 0.04  # 误差
                    print("unit_sleep:%s", unit_sleep)
                    self.send_cmd("STOP 0")
                    self.send_cmd("MD")
                    time.sleep(2)
                    self.send_cmd("STOP 2")
                    self.send_cmd("RROT " + str(rot_speed))
                    time.sleep(unit_sleep)
                    self.send_cmd("STOP 2")
                    self.send_cmd("MU")
                    time.sleep(2)
                    self.send_cmd("STOP 2")
                    self.redis.set(key,1,10*60*60)
                    # self.redis.set("is_working",0)
                    self.send_cmd("MF " + str(speed))
                    
                else:
                    print("work.py,vegetable position centery:{}".format(centery)) 
            else:
                print("work.py,vegetable has done,track_id:{}".format(track_id))
        
