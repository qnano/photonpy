# -*- coding: utf-8 -*-

import ctypes
import scipy.io
import numpy as np
import numpy.ctypeslib as ctl

from . import estimator
from .context import Context
from .lib import NullableFloatArrayType, NullableIntArrayType

Theta = ctypes.c_float * 5
FisherMatrix = ctypes.c_float * 16


class CSplineCalibration(ctypes.Structure):
    _fields_ = [
        ("n_voxels_z", ctypes.c_int32),
        ("n_voxels_y", ctypes.c_int32),
        ("n_voxels_x", ctypes.c_int32),
        ("z_min", ctypes.c_float),
        ("z_max", ctypes.c_float),
        ("coefs", ctl.ndpointer(np.float32, flags="aligned, c_contiguous"))
    ]

    def __init__(self, z_min, z_max, coefs):
        coefs = np.ascontiguousarray(coefs, dtype=np.float32)
        self.coefsx = coefs  # Hack to make sure the array doesn't get GCed

        self.n_voxels_x = coefs.shape[2]
        self.n_voxels_y = coefs.shape[1]
        self.n_voxels_z = coefs.shape[0]

        self.coefs = coefs.ctypes.data
        self.z_min = z_min
        self.z_max = z_max
        self.zrange = [z_min,z_max] # also available for other calibration objects
        ###

    @classmethod
    def from_file(cls, filename):
        if filename.endswith('.mat'):
            return cls.from_file_nmeth(filename)

        calibration = np.load(filename, allow_pickle=True)

        # Calibration file has z in nm, we use um
        z_min = calibration["zmin"] * 1e-3
        z_max = calibration["zmax"] * 1e-3

        coefs = calibration["coeff"]

        return cls(z_min, z_max, coefs)

    @classmethod
    def from_file_nmeth(cls, filename):
        mat = scipy.io.loadmat(filename)
        try:
            spline = mat["SXY"]['cspline'].item()
            coefs = spline['coeff'].item()
            dz = float(spline['dz'].item())
        except KeyError:
            spline = mat['cspline'].item()
            coefs = spline[0]
            dz = float(spline[1])

        # reading additional parameters from .mat file from Nature methods paper
        try:
            dz = float( mat['parameters']['dz'][0][0] )

        except KeyError:
            raise Exception( "Required parameters for C-spline can\'t be"
                             " found in the .mat file" )
        ###
        
        coefs=np.ascontiguousarray(coefs,dtype=np.float32)
        
        nx,ny,nz=coefs.shape[:-1]
        coefs = coefs.reshape((nx,ny,nz,4,4,4))
        
        # move the Z slices to the first axis and flip x and y
        # that the coefficients from nature methods PSF have ordering z,x,y, but voxels have x,y,z
        coefs = np.transpose(coefs, (2,0,1,3,5,4)).reshape((nz,ny,nx,64))

        #print(coefs.shape)
        nz = float(coefs.shape[0])
        z_min = -nz * dz * 1e-3 / 2
        z_max = (nz-1) * dz * 1e-3 / 2

        print(f"Z min={z_min}, max={z_max}. step size: {dz:.3f}")
        print(f"Voxels X:{coefs.shape[2]}, Y:{coefs.shape[1]}, Z:{coefs.shape[0]}",flush=True)
        return cls(z_min, z_max, coefs)


    def __str__( self ):
        """Print the parameters"""

        s = f'full z range = {self.z_range} um\n'
        s += f'z_min = {self.z_min} um\n'
        s += f'z_max = {self.z_max} um\n'
        s += f'n_voxels X = {self.n_voxels_x}\n'
        s += f'n_voxels Y = {self.n_voxels_y}\n'
        s += f'n_voxels Z = {self.n_voxels_z}\n'
        s += f'NMethods is used = {self.nmeth}\n'

        return s


class CSplinePSF(estimator.Estimator):
    def __init__(self, ctx:Context, inst, calib):
        super().__init__(ctx, inst, calib)
        
        
class CSplineMethods:
    
    FlatBg = 0
    TiltedBg = 1
    BgImage = 2
    
    def __init__(self, ctx: Context):
        smlmlib = ctx.smlm.lib
        self.ctx = ctx

        self._CSpline_CreatePSF = smlmlib.CSpline_CreatePSF
        self._CSpline_CreatePSF.argtypes = [
                ctypes.c_int32,
                ctypes.POINTER(CSplineCalibration),
                NullableFloatArrayType,
                ctypes.c_int32,# fitmode
                ctypes.c_bool,
                ctypes.c_void_p]
        self._CSpline_CreatePSF.restype = ctypes.c_void_p
        
        #// splineCoefs[shape[0]-3, shape[1]-3, shape[2]-3, 64]
        self._CSpline_Compute = smlmlib.CSpline_Compute 
        self._CSpline_Compute.argtypes = [
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"), # input shape
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), # input data
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), # output shape
            ctypes.c_int32 #border
            ]
        
        #void CSpline_Evaluate(const Int3& shape, const float* splineCoefs, const Vector3f& shift, int border, float* values, float* jacXYZ)
        self._CSpline_Evaluate = smlmlib.CSpline_Evaluate
        self._CSpline_Evaluate.argtypes = [
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"), # input shape
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), # splineCoefs
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), # shift [float3]
            ctypes.c_int32, #border
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous") # values
            ]
        
        #(const Int3& shape, const float* data, float* splineCoefs);

    def CreatePSF_XYZIBg(self, roisize, calib: CSplineCalibration, fitMode=FlatBg, cuda=True) -> CSplinePSF:
        inst = self._CSpline_CreatePSF(roisize, calib, None, fitMode, cuda, self.ctx.inst)
        return CSplinePSF(self.ctx, inst, calib)
    
    def ComputeSplineCoeff(self, data, border=0):
        """
        Compute spline coefficients for the given 3D matrix.
        The resulting set will have shape [*(shape-3+2),64]
        
        -3 because we go from grid to spline segments (4 points are needed to create 1 segment):
        0---1---2---3   ->   Spline from 1 to 2

        We add padding of one pixel on each side to clamp the outer values, resulting in +2
        """
        data = np.ascontiguousarray(data,dtype=np.float32)
        outshape = np.array(data.shape)-3+border*2
        spline = np.zeros((*outshape,64),dtype=np.float32)
        shape = np.ascontiguousarray(data.shape,dtype=np.int32)
        self._CSpline_Compute(shape, data, spline, border)
        return spline
    
    def Evaluate(self, splineCoefs, shift, border=0):
        """
        returns a matrix where the last dimension is value, dv/dx,dv/dy,dv/dz
        """
        assert len(splineCoefs.shape) == 4 and splineCoefs.shape[-1]==64
        splineCoefs = np.ascontiguousarray(splineCoefs,dtype=np.float32)
        shift = np.ascontiguousarray(shift, dtype=np.float32)
        outshape = np.array(splineCoefs.shape[:3]) - border*2
        values = np.zeros((*outshape,4), dtype=np.float32)
        self._CSpline_Evaluate(np.array(splineCoefs.shape,dtype=np.int32), splineCoefs, shift, border, values)
        return values
    
    def CreatePSFFromFile(self, roisize, filename, fitMode=FlatBg, cuda=True) -> CSplinePSF:
        
        if filename.endswith(".mat"):
            calib = CSplineCalibration.from_file_nmeth(filename)
            return self.CreatePSF_XYZIBg(roisize, calib, fitMode, cuda)

        elif filename.endswith(".pickle"):
            with open(filename, "rb") as f:
                import pickle
                d = pickle.load(f)
            return self.CreatePSFFromZStack(roisize, d['zstack'], d['zrange'], fitMode, cuda)

        elif filename.endswith(".zstack"):
            from photonpy.smlm.psf import load_zstack
            
            zstack, zrange = load_zstack(filename)
            
            return self.CreatePSFFromZStack(roisize, zstack, zrange, fitMode, cuda)
        else:
            raise RuntimeError(f"Unknown psf extension: {filename}")

    def CreatePSFFromZStack(self, roisize, zstack, zrange, fitMode=FlatBg, cuda=True, border=0) -> CSplinePSF:
        spline_coeff = self.ComputeSplineCoeff(zstack, border)
        calib = CSplineCalibration(zrange[0], zrange[-1], spline_coeff)
        calib.zstack = zstack
        return self.CreatePSF_XYZIBg(roisize, calib, fitMode, cuda)
    
