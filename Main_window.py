'''
author == Wenchen Chen
encoding  = utf-8
'''
import sys

import pyqtgraph

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

class Main_window(QWidget):
    def __init__(self):
        super(Main_window,self).__init__()
        self.iniUI()
        self.center_window()



    def iniUI(self):
        # 设置主窗口的标题
        self.setWindowTitle('Three mirror starting points')
        # 设置窗口图标


        # 设置窗口的尺寸
        self.resize(400,400)
        # 移动窗口位置
        label1 = QLabel('Half X FOV：')
        label2 = QLabel('Half Y FOV：')
        label3 = QLabel('EPD：')
        label4 = QLabel('S2 YDE：')
        label5 = QLabel('S2 ZDE：')
        label6 = QLabel('S4 YDE：')
        label7 = QLabel('S4 ZDE：')
        label8 = QLabel('SI YDE：')
        label9 = QLabel('SI ZDE：')
        line1 = QLineEdit()
        line1.setPlaceholderText('请输入X方向半视场，1°-15°')
        line2 = QLineEdit()
        line2.setPlaceholderText('请输入Y方向半视场，1°-15°')
        line3 = QLineEdit()
        line3.setPlaceholderText('请输入入瞳直径，12.5mm - 66mm')
        line4 = QLineEdit()
        line4.setPlaceholderText('请输入第一个曲面的Y方向偏心，50mm - 100mm')
        line5 = QLineEdit()
        line5.setPlaceholderText('请输入第一个面的Z方向偏心，100mm-350mm')
        line6 = QLineEdit()
        line6.setPlaceholderText('请输入第3个面的Y方向偏心，-100mm - 50mm')
        line7 = QLineEdit()
        line7.setPlaceholderText('请输入第3个面的Z方向偏心，150mm - 350mm')
        line8 = QLineEdit()
        line8.setPlaceholderText('请输入像面的Y方向偏心，-120mm - -20mm')
        line9 = QLineEdit()
        line9.setPlaceholderText('请输入像面的Z方向偏心，-80mm - 80mm')
        formlayout = QFormLayout()
        formlayout.addRow(label1, line1)
        formlayout.addRow(label2, line2)
        formlayout.addRow(label3, line3)
        formlayout.addRow(label4, line4)
        formlayout.addRow(label5, line5)
        formlayout.addRow(label6, line6)
        formlayout.addRow(label7, line7)
        formlayout.addRow(label8, line8)
        formlayout.addRow(label9, line9)
        self.setLayout(formlayout)



    def center_window(self):
        screen_size = QDesktopWidget().screenGeometry()
        window_size = self.geometry()
        left = (screen_size.width() - window_size.width())/2
        top = (screen_size.height() - window_size.height())/2
        self.move(left,top)

    def stop_program(self):
        app = QApplication.instance()
        # 退出应用程序
        app.quit()
    # def get_label_name(self,label_name):
    def onClick_Button(self):
        sender = self.sender()
        print(sender.text() + ' 按钮被按下')






if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = Main_window()

    window.show()

    sys.exit(app.exec_())







