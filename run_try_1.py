'''
author == Wenchen Chen
encoding  = utf-8
'''
import sys

import pyqtgraph

from PyQt5.QtWidgets import *
import sssss


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = sssss.Ui_mainWindow()
    # 向主窗口上添加控件
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())
