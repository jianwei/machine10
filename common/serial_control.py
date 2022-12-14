#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import json
import re
import serial
import time
import termios
from common.log import log


class serial_control():
    def __init__(self):
        port = "/dev/ttyACM0"  # Arduino端口
        # port = "/dev/tty.usbmodem1421301"  # Arduino端口
        # self.l = log(logfile="./serial_control.log")
        # self.l = log("~/serial_control.log")
        # self.logger = self.l.getLogger()
        self.timeout = 0.005

        f = open(port)
        attrs = termios.tcgetattr(f)
        attrs[2] = attrs[2] & ~termios.HUPCL
        termios.tcsetattr(f, termios.TCSAFLUSH, attrs)
        f.close()

        self.ser = serial.Serial()
        self.ser.baudrate = 9600
        self.ser.port = port
        self.ser.open()

    def close(self):
        self.ser.close()

    def send_cmd(self, message):
        ret = -2
        if ("cmd" in message.keys()):
            cmd = message["cmd"]
        else:
            cmd = None
        uuid = message["uuid"]
        # print("cmd:",cmd)
        if (cmd):
            # self.logger.info("cmd:%s,begin_time:%s", cmd, time.time())
            self.ser.write(cmd.encode())
            # self.logger.info("cmd:end write:%s", time.time())
            try:
                cnt = 1
                ret_all = ""
                time0 = time.time()
                while True:
                    cnt += 1
                    time1 = float(time.time())
                    response = self.ser.read()
                    # print("response:",response)
                    time2 = float(time.time())
                    diff = time2-time1
                    if (response):
                        ret_all += str(response, "UTF-8")
                        response_arr = ret_all.splitlines()
                        ret = response_arr[len(
                            response_arr)-1] if len(response_arr) > 0 else ""
                        # self.logger.info("1--cnt:%s,send_cmd:uuid:%s,cmd:%s,ret:%s,difftime:%s,response:%s", cnt, uuid, cmd, ret, diff, ret_all)
                        # time.sleep(0.1)
                        s1 = re.compile('^(-?[1-9]|0{1}\d*)$')
                        r1 = s1.findall(ret)
                        if (len(r1) > 0):
                            print("send_cmd:uuid:{},cmd:{},ret:{},difftime:{},response:{}".format(uuid, cmd, ret, diff, ret_all))
                            ret_dict = {
                                "uuid": uuid,
                                "cmd": cmd,
                                "retsult": ret,

                            }
                            self.ret_dict = ret_dict
                            print("break,cmd:{},end_time:{},ret_all:{}".format(cmd, time.time(), ret_all))
                            return ret
                        time3 = time.time()
                        if (time3-time0 >= 10):
                            print("break,time out")
                            break
            except Exception as e:
                print("serial连接或者执行失败,reason:", e)

    def get_ret(self):
        # print("ret_dict:",self.ret_dict)
        return self.ret_dict
