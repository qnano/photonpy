# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'linklocs.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(512, 370)
        self.btnEstimate = QtWidgets.QPushButton(Dialog)
        self.btnEstimate.setGeometry(QtCore.QRect(20, 40, 91, 31))
        self.btnEstimate.setObjectName("btnEstimate")
        self.txtLocsFile = QtWidgets.QLineEdit(Dialog)
        self.txtLocsFile.setGeometry(QtCore.QRect(130, 10, 261, 20))
        self.txtLocsFile.setObjectName("txtLocsFile")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(20, 10, 101, 21))
        self.label.setObjectName("label")
        self.maxDistance = QtWidgets.QDoubleSpinBox(Dialog)
        self.maxDistance.setGeometry(QtCore.QRect(320, 40, 62, 22))
        self.maxDistance.setProperty("value", 0.5)
        self.maxDistance.setObjectName("maxDistance")
        self.frameskip = QtWidgets.QSpinBox(Dialog)
        self.frameskip.setGeometry(QtCore.QRect(320, 70, 42, 22))
        self.frameskip.setProperty("value", 4)
        self.frameskip.setObjectName("frameskip")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(130, 40, 161, 21))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(130, 70, 111, 21))
        self.label_3.setObjectName("label_3")
        self.btnBrowse = QtWidgets.QPushButton(Dialog)
        self.btnBrowse.setGeometry(QtCore.QRect(400, 10, 71, 21))
        self.btnBrowse.setObjectName("btnBrowse")
        self.framePlot = QtWidgets.QFrame(Dialog)
        self.framePlot.setGeometry(QtCore.QRect(20, 100, 481, 261))
        self.framePlot.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.framePlot.setFrameShadow(QtWidgets.QFrame.Raised)
        self.framePlot.setObjectName("framePlot")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Estimate molecule on-time"))
        self.btnEstimate.setText(_translate("Dialog", "Estimate"))
        self.label.setText(_translate("Dialog", "Localizations (HDF5)"))
        self.label_2.setText(_translate("Dialog", "Max distance (in XY, pixels)"))
        self.label_3.setText(_translate("Dialog", "Max frames skipped"))
        self.btnBrowse.setText(_translate("Dialog", "Browse"))

