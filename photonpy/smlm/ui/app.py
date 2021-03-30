
import sys,os

import numpy as np
import tqdm
from PyQt5.QtWidgets import QApplication, QLineEdit, QFileDialog, QDialog,QVBoxLayout,QMessageBox,QCheckBox
from PyQt5 import QtGui
from PyQt5 import QtCore, QtWidgets
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from photonpy.smlm.dataset import Dataset
from photonpy.smlm.ui import main_ui, linklocs_ui
from photonpy.smlm.ui.progressbar import ProgressBar
from photonpy.smlm.ui.qtplot import PlotDialog

import threading
import json
import functools
import pyqtgraph as pg

import photonpy.smlm.process_movie as process_movie
import photonpy.smlm.extract_rois as extract_rois

from photonpy.smlm.util import imshow_hstack
from photonpy.smlm.ui.drift_correct_dlg import DriftCorrectionDialog



import matplotlib as mpl
#mpl.use('svg')
new_rc_params = {
#    "font.family": 'Times',
    "font.size": 15,
    "font.serif": [],
    "svg.fonttype": 'none'} #to store text as text, not as path
mpl.rcParams.update(new_rc_params)

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


#import photonpy.simflux.locs_to_pattern as simflux_pattern



#ript(Run In Plotting Thread) decorator
def ript(function):
    def ript_this(*args, **kwargs):
        global send_queue, return_queue, plot_thread
        if threading.currentThread() == plot_thread: #if called from the plotting thread -> execute
            return function(*args, **kwargs)
        else: #if called from a diffrent thread -> send function to queue
            send_queue.put(functools.partial(function, *args, **kwargs))
            return_parameters = return_queue.get(True) # blocking (wait for return value)
            return return_parameters
    return ript_this

def showMessage(txt):
   msg = QMessageBox()
   msg.setIcon(QMessageBox.Information)
   msg.setText(txt)
   msg.exec_()

def createDatasetViewer(ds:Dataset):    
    # Interpret image data as row-major instead of col-major
    pg.setConfigOptions(imageAxisOrder='row-major')
    
    img = ds.renderGaussianSpots(10, 0.5)
        
    ## Create window with ImageView widget
    win = QtGui.QDialog()
    win.resize(800,800)
    layout = QVBoxLayout(win)
    imv = pg.ImageView()
    layout.addWidget(imv)
    #win.setCentralWidget(imv)
    win.show()
    name = ds['locs_path']
    win.setWindowTitle(f'Viewing {name}')
        
    ## Add time-varying signal
    """
    sig = np.zeros(data.shape[0])
    sig[30:] += np.exp(-np.linspace(1,10, 70))
    sig[40:] += np.exp(-np.linspace(1,10, 60))
    sig[70:] += np.exp(-np.linspace(1,10, 30))
    
    sig = sig[:,np.newaxis,np.newaxis] * 3
    data[:,50:60,30:40] += sig 
    """
    
    imv.setImage(img)
    
    
    ## Display the data and assign each frame a time value from 1.0 to 3.0
    #imv.setImage(data, xvals=np.linspace(1., 3., data.shape[0]))
    
    ## Set a custom color map
    colors = [
        (0, 0, 0),
        (45, 5, 61),
        (84, 42, 55),
        (150, 87, 60),
        (208, 171, 141),
        (255, 255, 255)
    ]
    cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
    imv.setColorMap(cmap)
        
    return win

class LinkLocsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.ui = linklocs_ui.Ui_Dialog()
        self.ui.setupUi(self)

        self.ui.btnBrowse.clicked.connect(self._onBrowse)
        self.ui.btnEstimate.clicked.connect(self.estimate)
    
    def setLocsFile(self,fn):
        self.ui.txtLocsFile.setText(fn)

    def _onBrowse(self):
        options = QFileDialog.Options()
#        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"", "","All Files (*);;HDF5 Files (*.hdf5)", options=options)
        if fileName:
            self.ui.txtLocsFile.setText(fileName)
            
    def estimate(self):
        from utils.link_locs import estimate_on_time
        maxdist = self.ui.maxDistance.value()
        frameskip = self.ui.frameskip.value()
        fig,bins,framecounts = estimate_on_time(self.ui.txtLocsFile.text(),maxdist,frameskip)

        import photonpy.smlm.ui.qtplot as qtplot 
        plotdlg=qtplot.PlotDialog(fig,self)
        plotdlg.setModal(True)
        plotdlg.show()

def getWidgetValues(widgets):
    d={}
    for w in widgets:
        if type(w) == QtWidgets.QDoubleSpinBox or type(w) == QtWidgets.QSpinBox:
            v = w.value()
        elif type(w) == QLineEdit:
            v = w.text()
        elif type(w) == QCheckBox:
            v = w.isChecked()
        else:
            continue
        d[w.objectName()] = v
    return d

def setWidgetValues(widgets,values):
    for w in widgets:
        if w.objectName() in values:
            v = values[w.objectName()]
            if type(w) == QtWidgets.QDoubleSpinBox or type(w) == QtWidgets.QSpinBox:
                w.setValue(v)
            elif type(w) == QLineEdit:
                w.setText(v)
            elif type(w) == QCheckBox:
                w.setChecked(v)
                
                        

        

class Window(QDialog):
    localizeDone = QtCore.pyqtSignal()
    localizeFailed = QtCore.pyqtSignal([str])
    roiExtractionDone = QtCore.pyqtSignal()
    
    datasets = []

    def __init__(self):
        super().__init__()
        self.title = 'Photonpy localization microscopy analysis toolbox'
        
        self.viewers = []

        self.ui = main_ui.Ui_Dialog()
        ui=self.ui
        ui.setupUi(self)
    
        ui.btnBrowseTiff.clicked.connect(self.onBrowseTiff)
        ui.btnLocalize.clicked.connect(self.localize)
        ui.btnLinkLocs.clicked.connect(self.linklocs)
        
        ui.btnBrowseCameraDarkFrames.clicked.connect(self.onBrowseCameraDarkFrames)
        ui.btnBrowseCameraLightFrames.clicked.connect(self.onBrowseCameraLightFrames)
        ui.btnBrowseROIs.clicked.connect(self.onBrowseROIFile)
        
        ui.btnRCC.clicked.connect(self.onDriftCorrectRCC)
        ui.btnMinEntropyDrift.clicked.connect(self.onDriftCorrectMinEntropy)
        
        ui.btnExtractROIs.clicked.connect(self.onExtractROIs)
                
        ui.checkBoxPerPixelCamCalib.toggled.connect(self.onPerPixelCamCalibChanged)
        self.onPerPixelCamCalibChanged()
        
        ui.btnViewSelected.clicked.connect(self.onViewSelected)
        ui.btnLoad.clicked.connect(self.onLoadLocs)
        
        self.localizeFailed.connect(self.onLocalizeFailed)
        self.localizeDone.connect(self.onLocalizeDone)
        self.roiExtractionDone.connect(self.onROIExtractionDone)
        
        self.cfgFile = os.path.dirname(__file__) + '/ui-cfg.json'
        self.cfgWidgets = {
                ui.roisize,
                ui.gain,
                ui.offset,
                ui.detectionThreshold,
                ui.pixelsize,
                ui.spotDetectionPSFSigma,
                ui.spinSigmaFitFramesPerBin,
                ui.tiffPath,
                ui.txtCameraDarkFrames,
                ui.txtCameraLightFrames,
                ui.startFrame,
                ui.maxLinkDistance,
                ui.maxLinkFrameskip,
                ui.txtROIFile,
                ui.roiExtractMinSpotFrames,
                ui.roiExtractSpotFrames,
                ui.roiExtractAppend,
                ui.maxLinkDistanceIntensity,
                ui.checkBoxPerPixelCamCalib,
                ui.spinSpotDetectorUseMeanImage,
                ui.spinNumFrames,
                ui.chiSquareThreshold,
                ui.spinSumFrames,
                ui.rccFramesPerBin,
                ui.minEntFramesPerBin,
                ui.minEntMaxSpots
                }
        self.load()
        
    @property
    def selectedDataset(self):
        idx = self.ui.listDatasets.currentIndex().row()
        return self.datasets[idx]
    
    def onViewSelected(self):
        ds = self.selectedDataset
        self.viewers.append(createDatasetViewer(ds))
        
    def onDriftCorrectRCC(self):
        fpb = self.ui.rccFramesPerBin.value()
        ds = self.selectedDataset.copy()
        drift = ds.estimateDriftRCC(framesPerBin=fpb, maxdrift=5)
        ds.applyDrift(drift)
        path = os.path.splitext( ds['imagefile'])[0]+"_undrifted_rcc.hdf5"
        ds.save(path)
        ds['locs_path'] = path
        self.datasets.append(ds)
        self.updateList()
        
    def onDriftCorrectMinEntropy(self):
        fpb = self.ui.minEntFramesPerBin.value()
        maxspots = self.ui.minEntMaxSpots.value()
                
        ds = self.selectedDataset.copy()
        path_noext = os.path.splitext( ds['locs_path'])[0]
        
        rcc_fpb = self.ui.rccFramesPerBin.value()
        
        coarseFPB = self.ui.minEntCoarseFPB.value()
        if coarseFPB==0:
            coarseFPB=None
            
        coarseSigmaM = self.ui.minEntCoarseSigmaMultiplier.value()
        sigma = ds.data.crlb.pos.mean(0) * coarseSigmaM

        drift, prec = ds.estimateDriftMinEntropy(framesPerBin=fpb, 
                                                 pixelsize = self.ui.pixelsize.value(),
                                                 maxdrift = 5, maxspots = maxspots, 
                                                 initializeWithRCC = ds.numFrames//rcc_fpb,
                                                 coarseFramesPerBin = coarseFPB, 
                                                 coarseSigma = sigma,
                                                 outputfn = path_noext+"_drift_dme")
        ds.applyDrift(drift)
        path = path_noext+"_undrifted_dme.hdf5"
        ds.save(path)
        ds['locs_path'] = path
        self.datasets.append(ds)
        self.updateList()
        
    def onPerPixelCamCalibChanged(self):
        v = self.ui.checkBoxPerPixelCamCalib.checkState()
        self.ui.offset.setEnabled(not v)
        self.ui.gain.setEnabled(not v)
        self.ui.txtCameraDarkFrames.setEnabled(v)
        self.ui.txtCameraLightFrames.setEnabled(v)

        
    def load(self):
        path = os.path.abspath(self.cfgFile)
        print(f"Loading UI state from {path}")
        if os.path.exists(self.cfgFile):
            with open(self.cfgFile,'r') as f:
                d = json.load(f)
                setWidgetValues(self.cfgWidgets,d)
        
    def save(self):
        d = getWidgetValues(self.cfgWidgets)
        with open(self.cfgFile,'w') as f:
            json.dump(d,f,indent=4)
        
    def closeEvent(self,event):
        self.save()
        
    def linklocs(self):
        dlg = LinkLocsDialog(self)
        dlg.setLocsFile(self.ui.smlmLocsFile.text())
        dlg.show()
        
    def updatePaths(self):
        tiff_path = self.ui.tiffPath.text()
        
    def onBrowseCameraDarkFrames(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,"Browse movie containing dark calibration:", "","All Files (*);;TIFF File (*.tif)", options=options)
        if fileName:
            self.ui.txtCameraDarkFrames.setText(fileName)

    def onBrowseCameraLightFrames(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,"Browse movie containing light frames for calibration:", "","All Files (*);;TIFF File (*.tif)", options=options)
        if fileName:
            self.ui.txtCameraLightFrames.setText(fileName)

    def onBrowseROIFile(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,"Browse ROI file", "","All Files (*);;TIFF File (*.tif)", options=options)
        if fileName:
            self.ui.txtROIFile.setText(fileName)

    def onBrowseTiff(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,"Browse TIFF", "","All Files (*);;TIFF File (*.tif)", options=options)
        if fileName:
            self.ui.tiffPath.setText(fileName)
            self.updatePaths()
            
    def onLoadLocs(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self,"Browse ROI file", "","Picasso compatible HDF5 (*.hdf5);;Thunderstorm CSV (*.csv)", options=options)
        if filename:
            try:
                ds = Dataset.load(filename)
                self.result = ds
                self.datasets = [ds]
                self.updateList()
            except ValueError as e:
                showMessage(f'Error: {str(e)}')
            
    def onExtractROIs(self):
        locs_fn = self.ui.smlmLocsFile.text()
        tiff_path = self.ui.tiffPath.text()
        rois_path = self.ui.txtROIFile.text()

        pbar = ProgressBar("Extracting ROIs and estimating spot background and intensity")
        def progress_update(msg,done):
            if msg is not None:
                pbar.setMsg.emit(msg)
            if done is not None:
                pbar.update.emit(done)
            return not pbar.abortPressed
        
        cfg = self.getConfig()

        cfg = {**cfg,
             'maxlinkdistXY': self.ui.maxLinkDistance.value(),
             'maxlinkdistI': self.ui.maxLinkDistanceIntensity.value(),
             'maxlinkframeskip': self.ui.maxLinkFrameskip.value()
             }
        
        maxroiframes = self.ui.roiExtractSpotFrames.value()
        minroiframes = self.ui.roiExtractMinSpotFrames.value()
        appendFrames = self.ui.roiExtractAppend.value()
        
        def process_thread():
            self.rois,self.roiframes = extract_rois.extract_rois(rois_path, tiff_path, cfg, minroiframes, 
                                      maxroiframes, appendFrames, locs_fn, progress_update)
            if not pbar.abortPressed:             
                self.roiExtractionDone.emit()
            
        t = threading.Thread(target=process_thread)
        t.start()
        pbar.show()
        
    def onViewROIs(self):
        rois_path = self.ui.txtROIFile.text()
        roidata = extract_rois.ROIData.load(rois_path)
        
        plt.figure()
        for k in range(20):
            imshow_hstack(roidata.frames[k])

    def updateList(self):
        model = QtGui.QStandardItemModel()
        self.ui.listDatasets.setModel(model)
        
        for d in self.datasets:
            item = QtGui.QStandardItem(f"{d['locs_path']} - {d.info()}")
            model.appendRow(item)
            


    def getConfig(self):
        offset = self.ui.offset.value()
        gain = self.ui.gain.value()

        if self.ui.checkBoxPerPixelCamCalib.isChecked():
            offset = self.ui.txtCameraDarkFrames.text()
            gain = self.ui.txtCameraLightFrames.text()
            
            if len(offset) == 0:
                showMessage('Need to provide movie with dark frames')
                return
            
            if len(gain) == 0:
                showMessage('Need to provide movie with light frames')
                return 
                
        cfg = {
                'roisize': self.ui.roisize.value(),
                'threshold': self.ui.detectionThreshold.value(),
                'sigmaframesperbin': self.ui.spinSigmaFitFramesPerBin.value(),
                'gain': gain,
                'maxframes': self.ui.spinNumFrames.value(),
                'offset': offset,
                'startframe': self.ui.startFrame.value(),
                'pixelsize': self.ui.pixelsize.value(),
                'spotdetectsigma': self.ui.spotDetectionPSFSigma.value(),
                'sumframes': self.ui.spinSumFrames.value()
            }
    
        chisq = self.ui.chiSquareThreshold.value()
        if chisq > 0 :
            cfg['maxchisq'] = chisq
        
        return cfg
        
    def localize(self):
        tiff_path = self.ui.tiffPath.text()
        if not os.path.exists(tiff_path):
            return
        
        cfg = self.getConfig()
        if cfg is None:
            return
        
        locs_fn = os.path.splitext(tiff_path)[0]+".hdf5"

        self.ui.labelLocsInfo.setText('')
        
        pbar = ProgressBar("Running spot detection and 2D Gaussian localization...")

        def progress_update(msg,done):
            if msg is not None:
                pbar.setMsg.emit(msg)
            if done is not None:
                pbar.update.emit(done)
            return not pbar.abortPressed
        
        def localize_thread():
            print (f"Localize thread: {threading.get_ident()}")
            try:
                self.localizer = process_movie.Localizer2D()
                self.localizer.process(tiff_path, cfg, locs_fn, progress_update)
                
                self.tiff_path = tiff_path
                if not pbar.abortPressed:
                    self.localizeDone.emit()
            except ValueError as e:
                self.localizeFailed.emit(str(e))
         
        if True:
            t = threading.Thread(target=localize_thread)
            t.start()
        else: #debug -- skip the threading
            self.localizer = process_movie.Localizer2D()
            self.localizer.process(tiff_path, cfg, locs_fn, progress_update)
            self.localizeDone.emit()
        
        pbar.show()
        
    @QtCore.pyqtSlot(str)
    def onLocalizeFailed(self, msg):
        showMessage(f'Error: {msg}')

    @QtCore.pyqtSlot()
    def onLocalizeDone(self):
        print("localize done")
        
        self.localizer.plotChiSquare()
        self.localizer.plotSigmaTimeSeries()
        self.localizer.plotIntensityHistogram()
        self.result = self.localizer.result
        #img = self.result.renderGaussianSpots(20, 1)
        #plt.figure()
        #plt.imshow(img)
        
        self.viewers.append (createDatasetViewer(self.result))
        
        if 'sigma' in self.result.dtypeEstim.fields:
            sx = self.result.data.estim.sigma[:,0]
            sy = self.result.data.estim.sigma[:,1]
            self.ui.psfSigmaX.setValue(np.median(sx))
            self.ui.psfSigmaY.setValue(np.median(sy))
            
            fig = plt.figure(figsize=(8,5))
            plt.hist([sx,sy],label=['Sigma X','Sigma Y'],range=(1,3),bins=100)
            plt.legend()
            plt.xlabel('PSF Sigma [pixels]')
            plt.show()
            #PlotDialog(fig).show()
                
        self.datasets = [ self.result ]
        self.updateList()

        #self.ui.labelLocsInfo.setText(self.datasets[0].info())
        

    @QtCore.pyqtSlot()
    def onROIExtractionDone(self):
        print("roi extraction done")
        
def run_ui():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
       
    wnd = Window()
    wnd.show()
    wnd.activateWindow()
    app.exec_()
    wnd = None
    #del tqdm    # prevent exception at exit about not being able to join thread
    del app     # prevent IPython+Qt issue https://github.com/spyder-ide/spyder/issues/2970

if __name__ == '__main__':
    print('Opening UI')
    run_ui()