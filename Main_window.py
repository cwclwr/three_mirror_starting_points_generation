'''
author == Wenchen Chen
encoding  = utf-8
'''
import sys

import pyqtgraph
from PyQt5.QtCore import Qt
from PyQt5 import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from Generate_test_sys_parameter import *


class Main_window(QMainWindow):
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
        self.save_file_Bt = QPushButton('文件保存位置：')
        self.save_file_Bt.clicked.connect(self.msg)




        self.start_button = QPushButton('开始预测系统')
        self.start_button.clicked.connect(self.onClick_Button)
        self.stop_button = QPushButton('强制停止预测系统')
        self.stop_button.setIcon(QIcon('./images/Banshee.ico'))
        self.stop_button.clicked.connect(self.show_warnin_Dialog)







        self.checkbox = QCheckBox('预测系统需要进行优化')
        self.checkbox.stateChanged.connect(self.checkboxStatus)
        self.XFOV = QLineEdit()
        self.XFOV.setPlaceholderText('请输入X方向半视场，1°-15°')
        self.XFOV.editingFinished.connect(self.get_text)
        self.YFOV = QLineEdit()
        self.YFOV.setPlaceholderText('请输入Y方向半视场，1°-15°')
        self.EPD = QLineEdit()
        self.EPD.setPlaceholderText('请输入入瞳直径，12.5mm - 66mm')
        self.s2yde = QLineEdit()
        self.s2yde.setPlaceholderText('请输入第一个曲面的Y方向偏心，50mm - 100mm')
        self.s2zde = QLineEdit()
        self.s2zde.setPlaceholderText('请输入第一个面的Z方向偏心，100mm-350mm')
        self.s4yde = QLineEdit()
        self.s4yde.setPlaceholderText('请输入第3个面的Y方向偏心，-100mm - 50mm')
        self.s4zde = QLineEdit()
        self.s4zde.setPlaceholderText('请输入第3个面的Z方向偏心，150mm - 350mm')
        self.siyde = QLineEdit()
        self.siyde.setPlaceholderText('请输入像面的Y方向偏心，-120mm - -20mm')
        self.sizde = QLineEdit()
        self.sizde.setPlaceholderText('请输入像面的Z方向偏心，-80mm - 80mm')
        self.predict_sys_num = QLineEdit()
        self.predict_sys_num.editingFinished.connect(self.get_text)
        self.save_sys_num = QLineEdit()
        self.save_sys_num.editingFinished.connect(self.get_text)
        self.merit_box = QComboBox()
        self.merit_box.addItems(['Error function','RMS','Abs distortion','X_rel_distortion','Y_rel_distortion','s2 yde',
                           's4 yde','s2 s4 yde','s2 zde','s4 zde','s2 s4 zde','distance1','distance2','distance3',
                           'distance4','distance5','Volume','MTF'])

        self.merit_box.currentIndexChanged.connect(self.selection_change)
        self.save_file_line = QLineEdit()

        OKbutton = QPushButton('确认当前文件')
        OKbutton.clicked.connect(self.print_current_sys)
        OKbutton.clicked.connect(self.judge_system_parameters_appropriate)
        OKbutton.clicked.connect(self.show_status)

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
        hlayout1.addWidget(self.XFOV)
        hlayout2.addWidget(label2)
        hlayout2.addWidget(self.YFOV)
        hlayout3.addWidget(label3)
        hlayout3.addWidget(self.EPD)
        hlayout4.addWidget(label4)
        hlayout4.addWidget(self.s2yde)
        hlayout5.addWidget(label5)
        hlayout5.addWidget(self.s2zde)
        hlayout6.addWidget(label6)
        hlayout6.addWidget(self.s4yde)
        hlayout7.addWidget(label7)
        hlayout7.addWidget(self.s4zde)
        hlayout8.addWidget(label8)
        hlayout8.addWidget(self.siyde)
        hlayout9.addWidget(label9)
        hlayout9.addWidget(self.sizde)


        hlayout10.addWidget(label10)
        hlayout10.addWidget(self.predict_sys_num)
        hlayout11.addWidget(label11)
        hlayout11.addWidget(self.save_sys_num)
        hlayout12.addWidget(label12)
        hlayout12.addWidget(self.merit_box)

        hlayout13.addWidget(self.save_file_Bt)
        hlayout13.addWidget(self.save_file_line)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)

        # 设置第一列为垂直方向布局
        vlayout1.addStretch(2)
        vlayout1.addLayout(hlayout1)
        vlayout1.addLayout(hlayout2)
        vlayout1.addLayout(hlayout3)
        vlayout1.addLayout(hlayout4)
        vlayout1.addLayout(hlayout5)
        vlayout1.addLayout(hlayout6)
        vlayout1.addLayout(hlayout7)
        vlayout1.addLayout(hlayout8)
        vlayout1.addLayout(hlayout9)
        vlayout1.addWidget(OKbutton)
        vlayout1.addStretch(1)
        # 添加第二列
        vlayout2.addLayout(hlayout10)
        vlayout2.addLayout(hlayout11)
        vlayout2.addLayout(hlayout12)
        vlayout2.addLayout(hlayout13)
        vlayout2.addWidget(self.checkbox)
        vlayout2.addLayout(control_layout)


        main_layout.addLayout(vlayout1)
        main_layout.addLayout(vlayout2)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        main_frame = QWidget()
        main_frame.setLayout(main_layout)
        self.setCentralWidget(main_frame)




    def center_window(self):
        screen_size = QDesktopWidget().screenGeometry()
        window_size = self.geometry()
        left = (screen_size.width() - window_size.width())/2
        top = (screen_size.height() - window_size.height())/2
        self.move(left,top)

    def show_status(self):
        self.statusBar.showMessage('当前输入超出范围')

    def show_warnin_Dialog(self):
        messageBox = QMessageBox(QMessageBox.Warning, "Stop?", "stop the prediction program?")
        messageBox.setWindowIcon(QIcon(":/newPrefix/logo.ico"))
        Qyes = messageBox.addButton(self.tr("ok"), QMessageBox.YesRole)
        Qyes.clicked.connect(self.stop_program)
        Qno = messageBox.addButton(self.tr("cancel"), QMessageBox.NoRole)

        messageBox.setWindowModality(Qt.ApplicationModal)
        messageBox.exec_()

    def msg(self):
        openfile_name = QFileDialog.getExistingDirectory(None,"选取文件夹","C:/")  # 起始路径
        self.save_file_line.setText(openfile_name)
        print(openfile_name)



    def print_current_sys(self):

        XFOV = float(self.XFOV.text())
        YFOV = float(self.YFOV.text())
        EPD = float(self.EPD.text())
        total_sys_num = int(self.predict_sys_num.text())
        generarate_test_sys_parameters(XFOV=XFOV,YFOV=YFOV,EPD=EPD,total_sys_num=total_sys_num)
        current_sys = [XFOV,YFOV,EPD]

        print(current_sys)

    def judge_system_parameters_appropriate(self):
        print('xfov:',XFOV,'YFOV',YFOV,'EPD',EPD)
        if XFOV <= 15 and YFOV <= 15 and EPD <= 200/3 and (XFOV*4 + EPD) < (236/3+1e-5) \
                and (YFOV*4 + EPD) < (236/3+1e-5) and (XFOV + YFOV) < 25:
            pass
        else:
            print('the system parameters are beyond the boundary')
    def get_text(self):
        sender = self.sender()
        print(sender.text())

    def stop_program(self):
        app = QApplication.instance()
        app.quit()
    def onClick_Button(self):
        sender = self.sender()
        print(sender.text() + ' 按钮被按下')

    def selection_change(self):
        print('当前筛选标准为:',self.merit_box.currentText())



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
XFOV = 0
YFOV = 0
EPD = 0

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Main_window()
    window.show()
    sys.exit(app.exec_())








