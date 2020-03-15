import sys
import visual
import pandas as pd
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

class Data_Import(QWidget):
    def __init__(self,parent=None,mode=None):
        super(Data_Import, self).__init__(parent)
        self.setWindowTitle("数据导入")
        self.resize(400,300)
        if mode==0:
            self.mode = 0

        layout = QVBoxLayout()
        self.source = QTextEdit()
        self.btn0 = QPushButton("本地文件导入")
        self.btn1 = QPushButton("从数据库导入")
        self.btn2 = QPushButton("NEXT")
        layout.addWidget(self.source)
        layout.addWidget(self.btn0)
        layout.addWidget(self.btn1)
        layout.addWidget(self.btn2)
        self.setLayout(layout)

        self.btn0.clicked.connect(self.open_file)
        self.btn1.clicked.connect(self.database_connect)
        self.btn2.clicked.connect(self.nextPage)

    def open_file(self):
        file = QFileDialog.getOpenFileName(self,"选择文件")
        self.source.append(file[0])
        if file[0]:
            data_set = pd.read_csv(file[0])
            # global data
            # data = data_set
            # print(data_set)
        #关闭当前窗口，跳转至下一个窗口
        #self.hide()
        #self.next = visual.Visual()
        #self.next.show()
        self.dataset = data_set
        # global data
        # data = data_set



    def database_connect(self):
        pass

    def nextPage(self):

        #self.hide()
        self.next = visual.Visual()
        self.next.show()

if __name__=="__main__":
    app = QApplication(sys.argv)
    data = Data_Import()
    data.show()
    sys.exit(app.exec())