'''
author == Wenchen Chen
encoding  = utf-8
'''
import sys

import pyqtgraph

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

class Main_window(QMainWindow):
    def __init__(self):
        super(Main_window,self).__init__()
        self.iniUI()
    def iniUI(self):
        # 设置主窗口的标题
        self.setWindowTitle('Three mirror starting points')
        # 设置窗口图标
        self.setWindowIcon(QIcon('./images/Basilisk.ico'))

        # 设置窗口的尺寸
        self.resize(800,800)
        # 移动窗口位置

        self.move(300,300)
        self.center_window()



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
    app.setWindowIcon(QIcon('./images/Dragon.ico'))

    window = Main_window()
    window.center_window()

    window.show()

    sys.exit(app.exec_())







