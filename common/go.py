import json
import time
import numpy
from common.unix_socket import unix_socket
import functools


def cmpy(a, b):
    return b.get("centery")-a.get("centery")

def cmpx(a, b):
    return a.get("centerx")-b.get("centerx")

class go ():
    def __init__(self, redis):
        self.redis = redis
        self.default_machine_speed = 12
        self.current_machine_speed = 20
        self.increment = 2     # 速度增量
        self.min_unit_px = 5  # 每秒行驶多少像素
        self.last_check_time = float(time.time())
        self.last_turn_time = float(time.time())
        self.global_angle = 90
        self.line_cnt=1
       
    
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
    
    def set_current_speed(self):
        print("go.py set_default_speed")
        self.send_comand("MF "+str(self.current_machine_speed))

    def turn(self):
        pass

    def is_add_speed(self, redis_key):
        # print("redis_key",redis_key)
        data = self.redis.get(redis_key)
        if (data ):
            data = json.loads(data)
            if(len(data)>=2):
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
                    # print("-------------------------------------------------------------------------------------------------------")
                    # print("last_diff_time:{},difftime:{},diff_px:{},diff_px/diff_time:{}".format(last_diff_time,difftime,diff_px,diff_px/diff_time))
                    # print("-------------------------------------------------------------------------------------------------------")
                    if (float(time.time()) - self.last_check_time >0.2) :
                        self.last_check_time  = now 
                        if (abs(diff_px/diff_time) < self.min_unit_px):
                            self.current_machine_speed += self.increment
                            self.current_machine_speed = self.current_machine_speed if  self.current_machine_speed <=20 else 20
                            self.send_comand("MF "+str(self.current_machine_speed))
                        else:
                            # self.current_machine_speed = self.default_machine_speed
                            self.send_comand("MF "+str(self.current_machine_speed))
                    else:
                        print("0.2秒内，不重复加速")
    



    def get_left_side(self):
        pass
    

    def trun_angle(self,avg_centerx,point):
        unit = 0.2115  # 1 pint 0.2115cm
        gap = 115  # cm 导航摄像头的视野盲区
        cmd = "" 
        centerx = avg_centerx
        
        centery = point["centery"]
        screenSize = point["screenSize"]
        # center_point = avg_centerx
        center_point = screenSize[0]/2
        diff_point_x = centerx-center_point
        diff_point_y = point["screenSize"][1]-centery
        print("avg_centerx:{},center_point:{},diff_point_y:{}".format(avg_centerx,center_point,diff_point_y))
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
                # cmd_prefix = "TR" if global_angle < target_angle else "TL"
            else:
                target_angle = 90+angle
                cmd_prefix = "TL"
                # cmd_prefix = "TR"
        else:
            if (centerx <= center_point):
                target_angle = 90-angle
                cmd_prefix = "TL"
                # cmd_prefix = "TR"
            else:
                target_angle = 90+angle
                cmd_prefix = "TL" if global_angle < target_angle else "TR"
                # cmd_prefix = "TR" if global_angle < target_angle else "TL"
        # print("target_angle,global_angle5",target_angle,global_angle)
        # print("target_angle:{},global_angle:{}".format(target_angle,global_angle))
        if (target_angle != global_angle):
            cmd = cmd_prefix + " " + str(abs(target_angle-global_angle))
            self.global_angle = target_angle
            print("send-cmd:", cmd)
            self.send_comand(cmd)
        else:
            print("send-cmd:none")

    def get_target_x (self,frame):
        total = 0
        for item in frame:
            total += float(item.get("centerx"))
        avg_centerx = total/len(frame)
        return avg_centerx


    def split_line(self,frame):
        pass

    def split_list(self,frame):
        all_list= []
        if (self.line_cnt!=1):
            # frame = data[0]
            # frame = first_frame
            frame.sort(key=functools.cmp_to_key(cmpx))
            screenSize = frame[0].get("screenSize")
            x_line = screenSize[0]
            each_x = int(x_line/self.line_cnt)
            all_list= []
            for i in range(self.line_cnt):
                all_list.append([])
            for item in frame:
                index = int(item.get("centerx") / each_x)
                # print("index:",item.get("centerx"),each_x,index)
                all_list[index].append(item)
            # for item in all_list:
                # print ("-----------:",item)
        else :
            all_list= [frame]
        return all_list

            
    




    # def turn(self,data):
    def turn(self,redis_key):
        # now = float(time.time())
        # if(now-self.last_turn_time<1):
        #     print("1秒内不重复转向:",self.last_turn_time,now)
        #     self.last_turn_time = now
        #     return
        data = self.redis.get(redis_key)
        if (data and len(data)>=2):
            data = json.loads(data)
            first_frame = data[0]
            first_frame.sort(key=functools.cmp_to_key(cmpy))
            point = first_frame[0]
            #每行1株
            if(self.line_cnt==1):  
                avg_centerx = self.get_target_x(first_frame)
                # self.trun_angle(avg_centerx,point)
            #每行>1株
            else:
                #双数
                lists = self.split_list(first_frame)
                index_center = int(self.line_cnt / 2)
                # print("index_center:",index_center)
                if(self.line_cnt % 2 == 0):
                    x_center1= lists[index_center]
                    x_center2= lists[index_center-1]
                    avg_centerx1 = self.get_target_x(x_center1)
                    avg_centerx2 = self.get_target_x(x_center2)
                    avg_centerx = int((avg_centerx1+avg_centerx2)/2)
                    # print("avg_centerx1:{},avg_centerx2:{},avg_centerx:{}".format(avg_centerx1,avg_centerx2,avg_centerx))
                #单数
                else:
                    x_center = lists[index_center]
                    avg_centerx = self.get_target_x(x_center)
            self.trun_angle(avg_centerx,point)
                
                





# if __name__ == "__main__":
#     print(int(2/3))
#     obj = go("redis")
#     mock_data = [
#         [
#             {"point": [[450, 311], [517, 311], [450, 393], [517, 393]], "id": 0, "name": "person", "time": 1662082814.425567, "screenSize": [640, 480], "centerx": 183.5, "centery": 352.0, "center": [483.5, 352.0]},
#             {"point": [[450, 311], [517, 311], [450, 393], [517, 393]], "id": 0, "name": "person", "time": 1662082814.425567, "screenSize": [640, 480], "centerx": 103.5, "centery": 352.0, "center": [483.5, 352.0]},
#             {"point": [[450, 311], [517, 311], [450, 393], [517, 393]], "id": 0, "name": "person", "time": 1662082814.425567, "screenSize": [640, 480], "centerx": 213.5, "centery": 352.0, "center": [483.5, 352.0]},
#             {"point": [[450, 311], [517, 311], [450, 393], [517, 393]], "id": 0, "name": "person", "time": 1662082814.425567, "screenSize": [640, 480], "centerx": 223.5, "centery": 352.0, "center": [483.5, 352.0]},
#             {"point": [[450, 311], [517, 311], [450, 393], [517, 393]], "id": 0, "name": "person", "time": 1662082814.425567, "screenSize": [640, 480], "centerx": 233.5, "centery": 352.0, "center": [483.5, 352.0]},
#             {"point": [[450, 311], [517, 311], [450, 393], [517, 393]], "id": 0, "name": "person", "time": 1662082814.425567, "screenSize": [640, 480], "centerx": 383.5, "centery": 352.0, "center": [483.5, 352.0]},
#             {"point": [[450, 310], [522, 310], [450, 393], [522, 393]], "id": 1, "name": "person", "time": 1662082814.305369, "screenSize": [640, 480], "centerx": 486.0, "centery": 351.5, "center": [486.0, 351.5]}, 
#             {"point": [[451, 311], [517, 311], [451, 393], [517, 393]], "id": 2, "name": "person", "time": 1662082814.0092382, "screenSize": [640, 480], "centerx": 484.0, "centery": 352.0, "center": [484.0, 352.0]} 
#         ],
#         [
#             {"point": [[450, 311], [517, 311], [450, 393], [517, 393]], "id": 3, "name": "person", "time": 1662082814.425567, "screenSize": [640, 480], "centerx": 483.5, "centery": 352.0, "center": [483.5, 352.0]},
#             {"point": [[450, 310], [522, 310], [450, 393], [522, 393]], "id": 4, "name": "person", "time": 1662082814.305369, "screenSize": [640, 480], "centerx": 486.0, "centery": 351.5, "center": [486.0, 351.5]}, 
#             {"point": [[451, 311], [517, 311], [451, 393], [517, 393]], "id": 5, "name": "person", "time": 1662082814.0092382, "screenSize": [640, 480], "centerx": 484.0, "centery": 352.0, "center": [484.0, 352.0]} 
#         ],
#         [
#             {"point": [[450, 311], [517, 311], [450, 393], [517, 393]], "id": 6, "name": "person", "time": 1662082814.425567, "screenSize": [640, 480], "centerx": 483.5, "centery": 352.0, "center": [483.5, 352.0]},
#             {"point": [[450, 310], [522, 310], [450, 393], [522, 393]], "id": 7, "name": "person", "time": 1662082814.305369, "screenSize": [640, 480], "centerx": 486.0, "centery": 351.5, "center": [486.0, 351.5]}, 
#             {"point": [[451, 311], [517, 311], [451, 393], [517, 393]], "id": 8, "name": "person", "time": 1662082814.0092382, "screenSize": [640, 480], "centerx": 484.0, "centery": 352.0, "center": [484.0, 352.0]} 
#         ]
#     ]
#     obj.turn(json.dumps(mock_data))