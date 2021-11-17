'''
author == Wenchen Chen
encoding  = utf-8
'''
import sys

import pyqtgraph

from PyQt5.QtWidgets import *

class Main_window(QMainWindow):
    def __init__(self):
        super(Main_window,self).__init__()
        # 设置主窗口的标题
        self.setWindowTitle('Three mirror starting points')
        # self.setWindowTitle('')

        # 设置窗口的尺寸
        self.resize(800,800)
        # 移动窗口位置

        self.move(300,300)

    # def center(self):
    #     # 获取屏幕坐标系
    #     screen = QDesktopWidget().screenGeometry()
    #     # 获取窗口坐标系
    #     size = self.geometry()
    #     newLeft = (screen.width() - size.width()) / 2
    #     newTop = (screen.height() - size.height()) / 2
    #     self.move(newLeft, newTop)

    def center_window(self):
        screen_size = QDesktopWidget().screenGeometry()
        window_size = self.geometry()
        left = (screen_size.width() - window_size.width())/2
        top = (screen_size.height() - window_size.height())/2
        self.move(left,top)

    # def addaction(self):
    #     self.actions = self.action

    def my_button(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = Main_window()
    window.center_window()

    window.show()

    sys.exit(app.exec_())







