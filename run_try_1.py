'''
author == Wenchen Chen
encoding  = utf-8
'''
import sys

import pyqtgraph

from PyQt5.QtWidgets import *
import sssss
import try_001


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = try_001.Ui_MainWindow()
    # 向主窗口上添加控件
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())
