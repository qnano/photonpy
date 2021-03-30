# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'drift_correct.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_DriftCorrectionDialog(object):
    def setupUi(self, DriftCorrectionDialog):
        DriftCorrectionDialog.setObjectName("DriftCorrectionDialog")
        DriftCorrectionDialog.resize(597, 396)
        self.label = QtWidgets.QLabel(DriftCorrectionDialog)
        self.label.setGeometry(QtCore.QRect(10, 10, 101, 21))
        self.label.setObjectName("label")
        self.framePlot = QtWidgets.QFrame(DriftCorrectionDialog)
        self.framePlot.setGeometry(QtCore.QRect(10, 100, 581, 291))
        self.framePlot.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.framePlot.setFrameShadow(QtWidgets.QFrame.Raised)
        self.framePlot.setObjectName("framePlot")
        self.locsFile = QtWidgets.QLineEdit(DriftCorrectionDialog)
        self.locsFile.setGeometry(QtCore.QRect(110, 10, 371, 20))
        self.locsFile.setObjectName("locsFile")
        self.btnBrowse = QtWidgets.QPushButton(DriftCorrectionDialog)
        self.btnBrowse.setGeometry(QtCore.QRect(490, 10, 91, 21))
        self.btnBrowse.setObjectName("btnBrowse")
        self.btnEstimate = QtWidgets.QPushButton(DriftCorrectionDialog)
        self.btnEstimate.setGeometry(QtCore.QRect(10, 70, 121, 21))
        self.btnEstimate.setObjectName("btnEstimate")
        self.driftbins = QtWidgets.QSpinBox(DriftCorrectionDialog)
        self.driftbins.setGeometry(QtCore.QRect(110, 40, 101, 21))
        self.driftbins.setMaximum(10000)
        self.driftbins.setProperty("value", 500)
        self.driftbins.setObjectName("driftbins")
        self.label_2 = QtWidgets.QLabel(DriftCorrectionDialog)
        self.label_2.setGeometry(QtCore.QRect(10, 40, 101, 21))
        self.label_2.setObjectName("label_2")
        self.btnCorrect = QtWidgets.QPushButton(DriftCorrectionDialog)
        self.btnCorrect.setGeometry(QtCore.QRect(140, 70, 111, 21))
        self.btnCorrect.setObjectName("btnCorrect")
        self.btnLoad = QtWidgets.QPushButton(DriftCorrectionDialog)
        self.btnLoad.setGeometry(QtCore.QRect(260, 70, 131, 21))
        self.btnLoad.setObjectName("btnLoad")

        self.retranslateUi(DriftCorrectionDialog)
        QtCore.QMetaObject.connectSlotsByName(DriftCorrectionDialog)

    def retranslateUi(self, DriftCorrectionDialog):
        _translate = QtCore.QCoreApplication.translate
        DriftCorrectionDialog.setWindowTitle(_translate("DriftCorrectionDialog", "Drift correction"))
        self.label.setText(_translate("DriftCorrectionDialog", "Localizations:"))
        self.btnBrowse.setText(_translate("DriftCorrectionDialog", "Browse"))
        self.btnEstimate.setText(_translate("DriftCorrectionDialog", "Estimate drift"))
        self.label_2.setText(_translate("DriftCorrectionDialog", "Bins:"))
        self.btnCorrect.setText(_translate("DriftCorrectionDialog", "Correct drift"))
        self.btnLoad.setText(_translate("DriftCorrectionDialog", "Load drift from file"))

