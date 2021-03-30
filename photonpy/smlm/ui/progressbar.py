# -*- coding: utf-8 -*-


from . import progressbar_ui as pbar_ui

from PyQt5.QtWidgets import QDialog
from PyQt5 import QtCore

class ProgressBar(QDialog):
    update = QtCore.pyqtSignal(float)
    setMsg = QtCore.pyqtSignal(str)
    
    def __init__(self, message="Processing..."):
        super().__init__()
        self.ui = pbar_ui.Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.progressBar.setValue(0)
        self.ui.label.setText(message)
        self.setModal(True)
        self.abortPressed=False
        self.ui.btnAbort.clicked.connect(self._onAbort)
        self.update.connect(self._setProgress)
        self.setMsg.connect(self._setMsg)
        
    @QtCore.pyqtSlot(str)
    def _setMsg(self,msg):
        self.ui.label.setText(msg)
        
    @QtCore.pyqtSlot(float)
    def _setProgress(self,done):
        self.ui.progressBar.setValue(int(done*100))

    def _onAbort(self):
        self.abortPressed=True
