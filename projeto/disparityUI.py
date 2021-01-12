# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'disparityUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from disparity_map import *


class DisparityUI(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.calculateButton = QtWidgets.QPushButton(self.centralwidget)
        self.calculateButton.setGeometry(QtCore.QRect(110, 510, 151, 41))
        self.calculateButton.setObjectName("calculateButton")
        self.imageContainer = QtWidgets.QLabel(self.centralwidget)
        self.imageContainer.setGeometry(QtCore.QRect(380, 0, 411, 551))
        self.imageContainer.setText("")
        self.imageContainer.setScaledContents(True)
        self.imageContainer.setObjectName("imageContainer")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(300, 0, 131, 551))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.winSize = QtWidgets.QLineEdit(self.centralwidget)
        self.winSize.setGeometry(QtCore.QRect(180, 170, 111, 31))
        self.winSize.setClearButtonEnabled(False)
        self.winSize.setObjectName("winSize")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(60, 170, 61, 31))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(60, 220, 71, 41))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(60, 310, 61, 31))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(60, 260, 61, 31))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(60, 360, 71, 31))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(60, 410, 71, 31))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(60, 460, 81, 31))
        self.label_8.setObjectName("label_8")
        self.maxDisp = QtWidgets.QLineEdit(self.centralwidget)
        self.maxDisp.setGeometry(QtCore.QRect(180, 260, 111, 31))
        self.maxDisp.setClearButtonEnabled(False)
        self.maxDisp.setObjectName("maxDisp")
        self.minDisp = QtWidgets.QLineEdit(self.centralwidget)
        self.minDisp.setGeometry(QtCore.QRect(180, 220, 111, 31))
        self.minDisp.setClearButtonEnabled(False)
        self.minDisp.setObjectName("minDisp")
        self.blockSize = QtWidgets.QLineEdit(self.centralwidget)
        self.blockSize.setGeometry(QtCore.QRect(180, 310, 111, 31))
        self.blockSize.setClearButtonEnabled(False)
        self.blockSize.setObjectName("blockSize")
        self.uniqueness = QtWidgets.QLineEdit(self.centralwidget)
        self.uniqueness.setGeometry(QtCore.QRect(180, 360, 111, 31))
        self.uniqueness.setClearButtonEnabled(False)
        self.uniqueness.setObjectName("uniqueness")
        self.speckleSize = QtWidgets.QLineEdit(self.centralwidget)
        self.speckleSize.setGeometry(QtCore.QRect(180, 410, 111, 31))
        self.speckleSize.setClearButtonEnabled(False)
        self.speckleSize.setObjectName("speckleSize")
        self.speckleRange = QtWidgets.QLineEdit(self.centralwidget)
        self.speckleRange.setGeometry(QtCore.QRect(180, 460, 111, 31))
        self.speckleRange.setClearButtonEnabled(False)
        self.speckleRange.setObjectName("speckleRange")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(60, 80, 71, 31))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(60, 30, 61, 31))
        self.label_10.setObjectName("label_10")
        self.leftImage = QtWidgets.QLineEdit(self.centralwidget)
        self.leftImage.setGeometry(QtCore.QRect(180, 30, 111, 31))
        self.leftImage.setClearButtonEnabled(False)
        self.leftImage.setObjectName("leftImage")
        self.rightImage = QtWidgets.QLineEdit(self.centralwidget)
        self.rightImage.setGeometry(QtCore.QRect(180, 80, 111, 31))
        self.rightImage.setClearButtonEnabled(False)
        self.rightImage.setObjectName("rightImage")
        self.browseLeft = QtWidgets.QPushButton(self.centralwidget)
        self.browseLeft.setGeometry(QtCore.QRect(300, 30, 51, 31))
        self.browseLeft.setObjectName("browseLeft")
        self.browseRight = QtWidgets.QPushButton(self.centralwidget)
        self.browseRight.setGeometry(QtCore.QRect(300, 80, 51, 31))
        self.browseRight.setObjectName("browseRight")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


        ################################################
        # acrescentando funcionalidades aos botoes

        self.browseLeft.clicked.connect(self.getLeftFile)
        self.browseRight.clicked.connect(self.getRightFile)
        self.calculateButton.clicked.connect(self.calculate)

        ###################################################

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Reconstrução 3D"))
        self.calculateButton.setText(_translate("MainWindow", "Calcular mapa de disparidade"))
        self.winSize.setPlaceholderText(_translate("MainWindow", "Win Size"))
        self.label_2.setText(_translate("MainWindow", "Win Size:"))
        self.label_3.setText(_translate("MainWindow", "Min Disp:"))
        self.label_4.setText(_translate("MainWindow", "Block Size:"))
        self.label_5.setText(_translate("MainWindow", "Max Disp:"))
        self.label_6.setText(_translate("MainWindow", "Uniqueness:"))
        self.label_7.setText(_translate("MainWindow", "Speckle Size:"))
        self.label_8.setText(_translate("MainWindow", "Speckle Range:"))
        self.maxDisp.setPlaceholderText(_translate("MainWindow", "Max Disp"))
        self.minDisp.setPlaceholderText(_translate("MainWindow", "Min Disp"))
        self.blockSize.setPlaceholderText(_translate("MainWindow", "Block Size"))
        self.uniqueness.setPlaceholderText(_translate("MainWindow", "Uniqueness Ratio"))
        self.speckleSize.setPlaceholderText(_translate("MainWindow", "Speckle Window Size"))
        self.speckleRange.setPlaceholderText(_translate("MainWindow", "Speckle Range"))
        self.label_9.setText(_translate("MainWindow", "Right Image:"))
        self.label_10.setText(_translate("MainWindow", "Left Image:"))
        self.leftImage.setPlaceholderText(_translate("MainWindow", "Left Image"))
        self.rightImage.setPlaceholderText(_translate("MainWindow", "Right Image"))
        self.browseLeft.setText(_translate("MainWindow", "..."))
        self.browseRight.setText(_translate("MainWindow", "..."))

    #######################################################################################################
    # FUNCOES QUE MANEJAM AS ACOES DOS BOTOES

    def getLeftFile(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(None, 'Open file', 'c:\\',"Image files (*.jpg *.gif *.png)")
        if fname[0] == '':
            pass
        else:
            self.leftImage.setText(fname[0])

    def getRightFile(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(None, 'Open file', 'c:\\',"Image files (*.jpg *.gif *.png)")
        if fname[0] == '':
            pass
        else:
            self.rightImage.setText(fname[0])

    def calculate(self):
       li = self.leftImage.text()
       ri = self.rightImage.text()
       generateDisparityMap(li,ri)
       self.imageContainer.setPixmap(QtGui.QPixmap('disparity_map.jpg'))

    ########################################################################

def runDisparityUI():
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = DisparityUI()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
