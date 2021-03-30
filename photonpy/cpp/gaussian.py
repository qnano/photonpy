import ctypes
import numpy as np
import numpy.ctypeslib as ctl
import os,yaml

from photonpy.cpp.context import Context
from photonpy.cpp.estimator import Estimator


Theta = ctypes.c_float * 4
FisherMatrix = ctypes.c_float * 16



class Gauss3D_Calibration(ctypes.Structure):
    _fields_ = [
            ("x", ctypes.c_float * 4), 
            ("y", ctypes.c_float * 4),
            ("zrange", ctypes.c_float*2)
        ]

    def __init__(self, x=[1.3,2,3,0], y=[1.3,-2,3,0], zrange=[-3, 3]):
        self.x = (ctypes.c_float * 4)(*x)
        self.y = (ctypes.c_float * 4)(*y)
        self.zrange = (ctypes.c_float*2)(*zrange)
        
    def sigma(self):
        return self.x[0],self.y[0]

    @classmethod
    def from_file(cls, filename):
        ext = os.path.splitext(filename)[1]
        if ext == '.npy':
            calibration = np.load(filename, allow_pickle=True).item()
        else:
            with open(filename, "r") as f:
                calibration = yaml.load(f)
        return cls(calibration.get("x"), calibration.get("y"))

    def save(self, filename):
        ext = os.path.splitext(filename)[1]
        if ext == '.npy':
            np.save(filename, self.__dict__())
        elif ext == '.yaml':
            with open(filename, "w") as f:
                yaml.dump(self.__dict__(),f)
                
    def __dict__(self):
        return {'x': np.array(self.x,dtype=float).tolist(), 
                               'y': np.array(self.y,dtype=float).tolist(),
                               'range': np.array(self.zrange,dtype=float).tolist()}

class GaussianPSFMethods:
    def __init__(self, ctx: Context):
        smlmlib = ctx.smlm.lib
        self.ctx = ctx
        self.lib = ctx.smlm

        # CDLL_EXPORT void Gauss2D_EstimateIntensityBg(const float* imageData, Vector2f *IBg, int numspots, const Vector2f* xy,
        # const float *sigma, int imgw, int maxIterations, bool cuda)
        self._EstimateIBg = smlmlib.Gauss2D_EstimateIntensityBg
        self._EstimateIBg.argtypes = [
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # images
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # IBg (result)
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # IBg_crlb (result)
            ctypes.c_int32,  # numspots
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # xy (input)
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # roipos
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # sigma
            ctypes.c_int32,  # imgw
            ctypes.c_int32,  # maxiterations
            ctypes.c_int32  # cuda
        ]
                 
        # (float * image, int imgw, int imgh, float * spotList, int nspots)
        self._Gauss2D_Draw = smlmlib.Gauss2D_Draw
        self._Gauss2D_Draw.argtypes = [
            ctl.ndpointer(np.float64, flags="aligned, c_contiguous"),  # mu
            ctypes.c_int32,
            ctypes.c_int32,
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # mu
            ctypes.c_int32,
            ctypes.c_float
        ]

        self._Gauss2D_CreatePSF_XYIBg = smlmlib.Gauss2D_CreatePSF_XYIBg
        self._Gauss2D_CreatePSF_XYIBg.argtypes = [
                ctypes.c_int32,  # roisize
                ctypes.c_float,  # sigmax
                ctypes.c_float,  # sigmay
                ctypes.c_int32,  # cuda
                ctypes.c_void_p] # context
        self._Gauss2D_CreatePSF_XYIBg.restype = ctypes.c_void_p
        
        
        self._Gauss2D_CreatePSF_XYIBgConstSigma = smlmlib.Gauss2D_CreatePSF_XYIBgConstSigma
        self._Gauss2D_CreatePSF_XYIBgConstSigma.argtypes = [
                ctypes.c_int32,  # roisize
                ctypes.c_int32,  # cuda
                ctypes.c_void_p] # context
        self._Gauss2D_CreatePSF_XYIBgConstSigma.restype = ctypes.c_void_p

        self._Gauss2D_CreatePSF_XYITiltedBg = smlmlib.Gauss2D_CreatePSF_XYITiltedBg
        self._Gauss2D_CreatePSF_XYITiltedBg.argtypes = [
                ctypes.c_int32,  # roisize
                ctypes.c_int32,  # cuda
                ctypes.c_void_p] # context
        self._Gauss2D_CreatePSF_XYITiltedBg.restype = ctypes.c_void_p

        self._Gauss2D_CreatePSF_XYIBgSigma = smlmlib.Gauss2D_CreatePSF_XYIBgSigma
        self._Gauss2D_CreatePSF_XYIBgSigma.argtypes = [
                ctypes.c_int32,  # roisize
                ctypes.c_float, 
                ctypes.c_int32,  # cuda
                ctypes.c_void_p] # context
        self._Gauss2D_CreatePSF_XYIBgSigma.restype = ctypes.c_void_p
        
        self._Gauss2D_CreatePSF_XYIBgSigmaXY = smlmlib.Gauss2D_CreatePSF_XYIBgSigmaXY
        self._Gauss2D_CreatePSF_XYIBgSigmaXY.argtypes = [
                ctypes.c_int32,  # roisize
                ctypes.c_float, 
                ctypes.c_float, 
                ctypes.c_int32,  # cuda
                ctypes.c_void_p] # context
        self._Gauss2D_CreatePSF_XYIBgSigmaXY.restype = ctypes.c_void_p

        self._Gauss2D_CreatePSF_XYITiltedBgSigmaXY = smlmlib.Gauss2D_CreatePSF_XYITiltedBgSigmaXY
        self._Gauss2D_CreatePSF_XYITiltedBgSigmaXY.argtypes = [
                ctypes.c_int32,  # roisize
                ctypes.c_float, 
                ctypes.c_float, 
                ctypes.c_int32,  # cuda
                ctypes.c_void_p] # context
        self._Gauss2D_CreatePSF_XYITiltedBgSigmaXY.restype = ctypes.c_void_p

        self._Gauss2D_CreatePSF_XYZIBg = smlmlib.Gauss2D_CreatePSF_XYZIBg
        self._Gauss2D_CreatePSF_XYZIBg.argtypes = [
                ctypes.c_int32, 
                ctypes.POINTER(Gauss3D_Calibration), 
                ctypes.c_int32, 
                ctypes.c_void_p]
        self._Gauss2D_CreatePSF_XYZIBg.restype = ctypes.c_void_p

    # Spots is an array with rows: [ x,y, sigmaX, sigmaY, intensity ]
    def Draw(self, img, spots, addSigma=0):
        spots = np.ascontiguousarray(spots, dtype=np.float32)
        nspots = spots.shape[0]
        assert spots.shape[1] == 5
        img = np.ascontiguousarray(img, dtype=np.float64)
        self._Gauss2D_Draw(img, img.shape[1], img.shape[0], spots, nspots, addSigma)
        return img

    def EstimateIBg(self, images, sigma, xy, roipos=None, useCuda=False):
        images = np.ascontiguousarray(images, dtype=np.float32)
        xy = np.ascontiguousarray(xy, dtype=np.float32)
        assert len(images.shape) == 3
        numspots = len(images)
        assert np.array_equal( xy.shape, (numspots, 2))
        imgw = images.shape[1]
        result = np.zeros((numspots, 2), dtype=np.float32)
        crlb = np.zeros((numspots, 2), dtype=np.float32)

        sigma_ = np.zeros((numspots,2),dtype=np.float32)
        sigma_[:] = sigma
                
        if roipos is None:
            roipos = np.zeros((numspots,2))
        assert np.array_equal(roipos.shape, (numspots,2))
        roipos = np.ascontiguousarray(roipos,dtype=np.int32)
        if numspots > 0:
            self._EstimateIBg(images, result, crlb, numspots, xy, roipos, sigma_, imgw, 100, useCuda)
        return result, crlb
    
    # CDLL_EXPORT PSF* Gauss_CreatePSF_XYIBg(int roisize, float sigma, bool cuda);
    def CreatePSF_XYIBg(self, roisize, sigma, cuda) -> Estimator:
        """
        Create a PSF model with parameters [x,y,I,bg] (units of px,px,photons,photons/pixel).
        Sigma can be a tuple (x,y), indicating Gaussian shape in x and y direction.
        If sigma is None, the sigma's will be passed through as a per-spot constant
        """
        if sigma is not None:
            if np.isscalar(sigma):
                sigma_x, sigma_y = sigma,sigma
            else:
                sigma_x, sigma_y = sigma

            inst = self._Gauss2D_CreatePSF_XYIBg(roisize, sigma_x, sigma_y, cuda, self.ctx.inst)
        else:
            inst = self._Gauss2D_CreatePSF_XYIBgConstSigma(roisize, cuda, self.ctx.inst)
        return Estimator(self.ctx, inst, sigma)
    
    # CDLL_EXPORT PSF* Gauss_CreatePSF_XYIBg(int roisize, float sigma, bool cuda);
    def CreatePSF_XYITiltedBg(self, roisize, cuda) -> Estimator:
        """
        Create a PSF model with parameters [x,y,I,bg,bgx,bgy] (units of px,px,photons,photons/px^2,photons/px^3,photons/px^3).
        Model: mu = bg+bgx*(x-roisize*0.5)+bgy*(y-roisize*0.5)+I*Gaussian(x,y)
        Sigma can be a tuple (x,y), indicating Gaussian shape in x and y direction.
        """
        inst = self._Gauss2D_CreatePSF_XYITiltedBg(roisize, cuda, self.ctx.inst)
        return Estimator(self.ctx, inst)

    def CreatePSF_XYIBgSigma(self, roisize, initialSigma, cuda) -> Estimator:
        """
        Create a PSF model with parameters [x,y,I,bg,sigma] (units of px,px,photons,photons/pixel, px).
        InitialSigma is the initial value for sigma in the estimator.
        Model: mu = bg+I*Gaussian(x,y)
        All sigma's are clamped to 1 to improve robustness
        """
        inst = self._Gauss2D_CreatePSF_XYIBgSigma(roisize, initialSigma, cuda, self.ctx.inst)
        return Estimator(self.ctx, inst, initialSigma)

    def CreatePSF_XYIBgSigmaXY(self, roisize, initialSigma, cuda) -> Estimator:
        """
        Create a PSF model with parameters [x,y,I,bg,sigmaX,sigmaY] (units of px,px,photons,photons/pixel, px,px).
        InitialSigma is the initial value for sigma in the estimator.
        All sigma's are clamped to 1 to improve robustness
        """
        if np.isscalar(initialSigma):
            initialSigma=[initialSigma,initialSigma]
        inst = self._Gauss2D_CreatePSF_XYIBgSigmaXY(roisize, initialSigma[0], initialSigma[1], cuda, self.ctx.inst)
        return Estimator(self.ctx, inst, initialSigma)

    def CreatePSF_XYITiltedBgSigmaXY(self, roisize, initialSigma, cuda) -> Estimator:
        """
        Create a PSF model with parameters [x,y,I,bg,sigmaX,sigmaY] (units of px,px,photons,photons/pixel, px,px).
        InitialSigma is the initial value for sigma in the estimator.
        All sigma's are clamped to 1 to improve robustness
        """
        if np.isscalar(initialSigma):
            initialSigma=[initialSigma,initialSigma]
        inst = self._Gauss2D_CreatePSF_XYITiltedBgSigmaXY(roisize, initialSigma[0], initialSigma[1], cuda, self.ctx.inst)
        return Estimator(self.ctx, inst, initialSigma)

    # CDLL_EXPORT PSF* Gauss_CreatePSF_XYZIBg(int roisize, const Gauss3D_Calibration& calib, bool cuda);
    def CreatePSF_XYZIBg(self, roisize, calib: Gauss3D_Calibration, cuda) -> Estimator:
        inst = self._Gauss2D_CreatePSF_XYZIBg(roisize, calib, cuda, self.ctx.inst)
        return Estimator(self.ctx, inst, calib)


Gaussian = GaussianPSFMethods

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    calib = Gauss3D_Calibration()
    
    with Context() as ctx:
        sigma=1.5
        roisize=12
        theta=[[roisize//2, roisize//2, 3, 1000, 5]]
        g_api = GaussianPSF(ctx)
        psf = g_api.CreatePSF_XYZIBg(roisize, calib, True)
    
        imgs = psf.ExpectedValue(theta)
        plt.figure()
        plt.imshow(imgs[0])
        
        sample = np.random.poisson(imgs)
    
        # Run localization on the sample
        estimated,diag,traces = psf.Estimate(sample)        
            
        print(f"Estimated position: {estimated[0]}")
        
        
