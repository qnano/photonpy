from PyQt5.QtWidgets import QDialog,QFileDialog,QApplication,QHBoxLayout,QVBoxLayout
from PyQt5 import QtCore
import sys
from . import drift_correct_ui

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from photonpy.smlm.drift_estimate import rcc

class DriftCorrectionDialog(QDialog):
    
    def __init__(self, parent=None, locsFile=None):
        super().__init__(parent)
        self.ui = drift_correct_ui.Ui_DriftCorrectionDialog()
        self.ui.setupUi(self)
        self.setModal(True)
        
        if locsFile is not None:
            self.ui.locsFile.setText(locsFile)
        
        self.ui.btnBrowse.clicked.connect(self._onBrowse)
        self.ui.btnEstimate.clicked.connect(self._onEstimate)
        self.ui.btnCorrect.clicked.connect(self._onCorrect)
        
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        
        layout = QHBoxLayout()
        layout.addWidget(self.canvas)
        self.ui.framePlot.setLayout(layout)

        ax = self.figure_ax = self.figure.add_subplot(111)
        ax.clear()
        ax.plot(np.sin(np.linspace(0,4,100)))
        
        
    def _onBrowse(self):
        options = QFileDialog.Options()
#        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"", "","All Files (*);;HDF5 Files (*.hdf5)", options=options)
        if fileName:
            self.ui.locsFile.setText(fileName)
        
    def _onEstimate(self):
        drift = drift_estimate(self.ui.locsFile.text(),self.ui.driftbins.value())
        self.figure_ax.clear()
        #self.figure_ax.plot(drift)
        
        self.figure_ax.plot(drift[:, 0], label="X drift")
        self.figure_ax.plot(drift[:, 1], label="Y drift")
        #plt.xlabel("Timebins")
        #plt.ylabel("Pixels")
        self.canvas.draw()
        #plt.show()
#        plt.legend()
        
    def _onCorrect(self):
        ...



if __name__ == '__main__':
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
       
    wnd = DriftCorrectionDialog()
    wnd.show()
    wnd.activateWindow()
    app.exec_()
    del app     # prevent IPython+Qt issue https://github.com/spyder-ide/spyder/issues/2970
