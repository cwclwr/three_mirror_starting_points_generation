'''
author == Wenchen Chen
encoding  = utf-8
'''


from PyQt5.QtCore import *

class mysignal(QObject):
    sys_param_signal = pyqtSignal(list)
    structure_signal = pyqtSignal(list)
    def judge_appropriate(self,sys_param):
        if sys_param[0] > 15:
            print('the XFOV is beyond the boundary')
        if sys_param[1] > 15:
            print('the YFOV is beyond the boundary')
        if sys_param[2] > 200/3:
            print('the EPD is beyond the boundary')
        if sys_param[0]+ sys_param[1]>25:
            print('the full FOV is so large')
        if (4*sys_param[0] + sys_param[2]) > 236/3 or (4*sys_param[1] + sys_param[2])> 236/3 :
            print('the system parameter is beyond the boundary')
        if sys_param[0]<15 and sys_param[1]<15 and sys_param[2]<200/3 and  sys_param[0]+ sys_param[1] < 25 \
                           and (4*sys_param[0] + sys_param[2]) < 236/3 or (4*sys_param[1] + sys_param[2]) < 236/3:
            print('the system parameter is in the boundary')
system_signal =mysignal()


system_signal.sys_param_signal.connect(system_signal.judge_appropriate)

system_signal.sys_param_signal.emit([16,10,49])










class MyTypeSignal(QObject):
    # 定义一个信号
    sendmsg = pyqtSignal(object)

    sendmsg1 = pyqtSignal(str,int,int)

    def run(self):
        self.sendmsg.emit('Hello PyQt5')

    def run1(self):
        self.sendmsg1.emit("hello",3,4)


class MySlot(QObject):
    def get(self,msg):
        print("信息：" + msg)
    def get1(self,msg,a,b):
        print(msg)
        print(a+b)


if __name__ == '__main__':
    send = MyTypeSignal()
    slot = MySlot()

    send.sendmsg.connect(slot.get)
    send.sendmsg1.connect(slot.get1)


    send.run()
    send.run1()

    send.sendmsg.disconnect(slot.get)
    send.run()