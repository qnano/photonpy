
import sys
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import random

class PlotDialog(QDialog):
    def __init__(self, figure, parent=None):
        super(PlotDialog, self).__init__(parent)
        
        self.figure = figure

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.canvas.draw()
        
#        self.refresh()

    def refresh(self):
        data = [random.random() for i in range(10)]

        # create an axis
        ax = self.figure.add_subplot(111)

        # discards the old graph
        # ax.hold(False) # deprecated, see above

        # plot data
        ax.plot(data, '*-')
        
        # refresh canvas
        self.canvas.draw()
        

if __name__ == '__main__':
    app = QApplication(sys.argv)

    fig = plt.figure()
#    ax = fig.add_subplot(111)
 #   ax.plot(data, '*-')

    main = PlotDialog(fig)
    main.show()
    
    sys.exit(app.exec_())
    del app