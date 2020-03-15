
import data_import
import sys
from PyQt5 import QtWidgets,QtCore,QtGui

class Mode(QtWidgets.QMainWindow):
    def __init__(self,parent=None):
        super(Mode, self).__init__(parent)
        self.setWindowTitle("数据分析可视化系统")
        self.resize(400,300)

        modes = ['线性回归', '决策树', '聚类', '关联规则']
        self.btn1 = QtWidgets.QPushButton(modes[0])
        self.btn1.clicked.connect(self.lr)
        self.btn2 = QtWidgets.QPushButton(modes[1])
        self.btn3 = QtWidgets.QPushButton(modes[2])
        self.btn4 = QtWidgets.QPushButton(modes[3])

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.btn1,QtCore.Qt.AlignCenter)
        layout.setStretch(0, 0.7)
        layout.addWidget(self.btn2, QtCore.Qt.AlignCenter)
        layout.setStretch(1, 0.7)
        layout.addWidget(self.btn3, QtCore.Qt.AlignCenter)
        layout.setStretch(2, 0.7)
        layout.addWidget(self.btn4, QtCore.Qt.AlignCenter)
        layout.setStretch(3, 0.7)
        #self.setLayout(layout)

        frame = QtWidgets.QWidget()
        frame.setLayout(layout)
        self.setCentralWidget(frame)


    #线性回归
    def lr(self):
        self.next = data_import.Data_Import(mode=0) #跳转至数据导入，隐藏当前窗口
        self.next.show()
        #mainApp.hide()



if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainApp = Mode()
    mainApp.show()
    sys.exit(app.exec())