'''
author == Wenchen Chen
encoding  = utf-8
'''
import sys

import pyqtgraph
from PyQt5.QtCore import Qt

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


        # 设置窗口的尺寸
        self.resize(400,400)
        # 移动窗口位置
        label1 = QLabel(' X FOV：')
        label2 = QLabel(' Y FOV：')
        label3 = QLabel('   EPD：')
        label4 = QLabel('S2 YDE：')
        label5 = QLabel('S2 ZDE：')
        label6 = QLabel('S4 YDE：')
        label7 = QLabel('S4 ZDE：')
        label8 = QLabel('SI YDE：')
        label9 = QLabel('SI ZDE：')
        label10 = QLabel('预测系统个数：')
        label11 = QLabel('保留系统个数：')
        label12 = QLabel('   筛选指标： ')
        label13 = QLabel('文件保存位置：')
        start_button = QPushButton('开始预测系统')
        start_button.clicked.connect(self.onClick_Button)
        stop_button = QPushButton('强制停止预测系统')
        stop_button.setIcon(QIcon('./images/Banshee.ico'))
        stop_button.clicked.connect(self.onClick_Button)

        self.checkbox = QCheckBox('预测系统需要进行优化')
        self.checkbox.stateChanged.connect(self.checkboxStatus)



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
        line10 = QLineEdit()
        line11 = QLineEdit()
        self.line_box = QComboBox()
        self.line_box.addItems(['Error function','RMS','Abs distortion','X_rel_distortion','Y_rel_distortion','s2 yde',
                           's4 yde','s2 s4 yde','s2 zde','s4 zde','s2 s4 zde','distance1','distance2','distance3',
                           'distance4','distance5','Volume','MTF'])

        self.line_box.currentIndexChanged.connect(self.selection_change)
        line13 = QLineEdit()


        vlayout1 = QVBoxLayout()
        vlayout2 = QVBoxLayout()
        hlayout1 = QHBoxLayout()
        hlayout2 = QHBoxLayout()
        hlayout3 = QHBoxLayout()
        hlayout4 = QHBoxLayout()
        hlayout5 = QHBoxLayout()
        hlayout6 = QHBoxLayout()
        hlayout7 = QHBoxLayout()
        hlayout8 = QHBoxLayout()
        hlayout9 = QHBoxLayout()

        hlayout10 = QHBoxLayout()
        hlayout11 = QHBoxLayout()
        hlayout12 = QHBoxLayout()
        hlayout13 = QHBoxLayout()
        control_layout = QHBoxLayout()
        main_layout = QHBoxLayout()

        # 向第一列中添加数据
        hlayout1.addWidget(label1)
        hlayout1.addWidget(line1)
        hlayout2.addWidget(label2)
        hlayout2.addWidget(line2)
        hlayout3.addWidget(label3)
        hlayout3.addWidget(line3)
        hlayout4.addWidget(label4)
        hlayout4.addWidget(line4)
        hlayout5.addWidget(label5)
        hlayout5.addWidget(line5)
        hlayout6.addWidget(label6)
        hlayout6.addWidget(line6)
        hlayout7.addWidget(label7)
        hlayout7.addWidget(line7)
        hlayout8.addWidget(label8)
        hlayout8.addWidget(line8)
        hlayout9.addWidget(label9)
        hlayout9.addWidget(line9)


        hlayout10.addWidget(label10)
        hlayout10.addWidget(line10)
        hlayout11.addWidget(label11)
        hlayout11.addWidget(line11)
        hlayout12.addWidget(label12)
        hlayout12.addWidget(self.line_box)

        hlayout13.addWidget(label13)
        hlayout13.addWidget(line13)
        control_layout.addWidget(start_button)
        control_layout.addWidget(stop_button)

        # 设置第一列为垂直方向布局
        vlayout1.addLayout(hlayout1)
        vlayout1.addLayout(hlayout2)
        vlayout1.addLayout(hlayout3)
        vlayout1.addLayout(hlayout4)
        vlayout1.addLayout(hlayout5)
        vlayout1.addLayout(hlayout6)
        vlayout1.addLayout(hlayout7)
        vlayout1.addLayout(hlayout8)
        vlayout1.addLayout(hlayout9)
        # 添加第二列
        vlayout2.addLayout(hlayout10)
        vlayout2.addLayout(hlayout11)
        vlayout2.addLayout(hlayout12)
        vlayout2.addLayout(hlayout13)
        vlayout2.addWidget(self.checkbox)
        vlayout2.addLayout(control_layout)


        main_layout.addLayout(vlayout1)
        main_layout.addLayout(vlayout2)


        self.setLayout(main_layout)



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
    def onClick_Button(self):
        sender = self.sender()
        print(sender.text() + ' 按钮被按下')

    def selection_change(self):
        print('当前筛选标准为:',self.line_box.currentText())

    def checkboxStatus(self):
        global system_mode
        system_mode = 0

        if self.checkbox.checkState() == 2:
            system_mode = 2
            print('当前系统模式为：',system_mode)
        else:
            system_mode = self.checkbox.checkState()
            print('当前系统模式为：',system_mode)
        return system_mode

# system_mode = 0
# print('初始的system model为:',system_mode)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Main_window()
    window.show()
    sys.exit(app.exec_())








