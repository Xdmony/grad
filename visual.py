import sys

from qtpy.QtWidgets import *
from qtpy.QtCore import QUrl
from qtpy.QtWebEngineWidgets import QWebEngineView

class Visual(QMainWindow):

    def __init__(self,parent=None,f_path=None):
        super(Visual, self).__init__(parent)
        self.setWindowTitle("可视化系统")
        self.resize(1000,600)

        self.view = QWebEngineView()
        self.console = QGroupBox()

        main_frame = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(self.console)
        layout.addWidget(self.view)
        main_frame.setLayout(layout)

        self.setCentralWidget(main_frame)
        self.view.load(QUrl("skl/20200129中国疫情地图.html"))



        # controller = QGroupBox()
        # view = QWebEngineView()
        #
        # layout = QHBoxLayout()
        #
        # layout.addStretch(1)
        # layout.addWidget(controller)
        #
        # layout.addStretch(4)
        # layout.addWidget(view)
        #
        # view.load(QUrl("www.baidu.com"))
        # view.setVisible(True)
        # view.show()
        # self.setLayout(layout)

        # self.view = QWebEngineView()
        # self.view.load(QUrl("www.google.com"))
        # self.view.show()


if __name__=="__main__":
    app = QApplication(sys.argv)
    visual = Visual()
    visual.show()
    sys.exit(app.exec())