import sys

from qtpy.QtWidgets import QApplication,QWidget,QMainWindow
#from qtpy.QtWebEngineWidgets import QWebEngineView

class Visual(QMainWindow):
    def __init__(self,parent=None,f_path=None):
        super(Visual, self).__init__(parent)
        self.resize(800,600)





if __name__=="__main__":
    app = QApplication(sys.argv)
    visual = Visual()
    visual.show()
    sys.exit(app.exec())