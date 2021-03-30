""" 
Dataset class to manage localization dataset. Picasso compatible HDF5 and thunderstorm CSV are supported

photonpy - Single molecule localization microscopy library
© Jelmer Cnossen 2018-2021
"""# -*- coding: utf-8 -*-
import os
from photonpy import Context, PostProcessMethods, GaussianPSFMethods, Estimator
import numpy as np
from photonpy.smlm.frc import FRC
from photonpy.smlm.drift_estimate import minEntropy,rcc
import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline
from photonpy.utils.multipart_tiff import MultipartTiffSaver
from scipy.ndimage import median_filter

import h5py 
import yaml

class Dataset:
    """
    Keep a localization dataset using a numpy structured array
    """
    def __init__(self, length, dims, imgshape=None, data=None, config=None, haveSigma=False, extraFields=None, **kwargs):

        if imgshape is None:
            imgshape = np.array(np.ceil([np.max(data, i) for i in range(dims)]),dtype=np.int32)

        self.imgshape = np.array(imgshape)
        if data is not None:
            self.data = np.copy(data).view(np.recarray)
        else:
            dtype = self.createDTypes(dims, len(imgshape), haveSigma, extraFields)
            self.data = np.recarray(length, dtype=dtype)
            self.data.fill(0)

        self.sigma = np.zeros(dims)
        self.config = config if config is not None else {}
        
        if extraFields is not None:
            self.config['extraFields'] = extraFields
        
        if kwargs is not None:
            self.config = {**self.config, **kwargs}
                    
    @staticmethod
    def merge(datasets):
        ds1 = datasets[0]
        n = sum([len(ds) for ds in datasets])
        copy = type(ds1)(n, ds1.dims, ds1.imgshape, config=ds1.config)
        j = 0
        for i,ds in enumerate(datasets):
            copy.data[j:j+len(ds)] = ds.data
            j += len(ds)
        return copy
        
    def __add__(self, other):
        return Dataset.merge([self,other])

    def __getitem__(self,idx):
        if type(idx) == str:
            return self.config[idx]
        else:
            indices = idx
            return type(self)(0, self.dims, self.imgshape, self.data[indices], config=self.config)

    def __setitem__(self, idx, val):
        if type(idx) == str:
            self.config[idx]=val
        else:
            if not isinstance(val, Dataset):
                raise ValueError('dataset[val] __setitem__ operator expects another dataset')
            
            if len(val) != len(self.data[idx]):
                raise ValueError('dataset __setitem__ operator: Lengths do not match, expecting {len(self.data[idx])} received {len(val)}')
            
            self.data[idx] = val.data

    def copy(self):
        return self[np.arange(len(self))]
            
    def createDTypes(self,dims, imgdims, includeGaussSigma, extraFields=None):
        """
        Can be overriden to add columns
        """
        dtypeEstim = [
            ('pos', np.float32, (dims,)),
            ('photons', np.float32),
            ('bg', np.float32)]
        
        if includeGaussSigma:
            dtypeEstim.append(
                ('sigma', np.float32, (2,))
            )
                    
        self.dtypeEstim = np.dtype(dtypeEstim)
        
        dtypeLoc = [
            ('roi_id', np.int32),  # typically used as the original ROI index in an unfiltered dataset
            ('frame', np.int32),
            ('estim', self.dtypeEstim),
            ('crlb', self.dtypeEstim),
            ('chisq', np.float32),
            ('group', np.int32),
            ('roipos', np.int32, (imgdims,))
            ]

        if extraFields is not None:
            for f in extraFields:
                dtypeLoc.append(f)
        
        return np.dtype(dtypeLoc)

    def hasPerSpotSigma(self):
        return 'sigma' in self.data.dtype.fields
            
    def filter(self, indices):
        """
        Keep only the selected indices, filtering in-place. An alternative is doing a copy: ds2 = ds[indices]
        """
        prevcount = len(self)
        self.data = self.data[indices]
        
        print(f"Keeping {len(self.data)}/{prevcount} spots")
    
    
    
    @property
    def numFrames(self):
        if len(self) == 0:
            return 0
        
        return np.max(self.data.frame)+1
            
    def indicesPerFrame(self):
        frame_indices = self.data.frame
        if len(frame_indices) == 0: 
            numFrames = 0
        else:
            numFrames = np.max(frame_indices)+1
        frames = [[] for i in range(numFrames)]
        for k in range(len(self.data)):
            frames[frame_indices[k]].append(k)
        for f in range(numFrames):
            frames[f] = np.array(frames[f], dtype=int)
        return frames
            
    def __len__(self):
        return len(self.data)
    
    @property
    def dims(self):
        return self.data.estim.pos.shape[1]
    
    @property
    def pos(self):
        return self.data.estim.pos
    
    @pos.setter
    def pos(self, val):
        self.data.estim.pos = val
    
    @property
    def crlb(self):
        return self.data.crlb
    
    @property
    def photons(self):
        return self.data.estim.photons

    @photons.setter
    def photons(self, val):
        self.data.estim.photons = val
    
    @property
    def background(self):
        return self.data.estim.bg

    @background.setter
    def background(self,val):
        self.data.estim.bg = val
    
    @property
    def frame(self):
        return self.data.frame
    
    @property
    def chisq(self):
        return self.data.chisq
    
    @chisq.setter
    def chisq(self,val):
        self.data.chisq = val
    
    @property
    def roi_id(self):
        return self.data.roi_id
    
    @property
    def group(self):
        return self.data.group
    
    @group.setter
    def group(self,val):
        self.data.group = val
    
    @property
    def local_pos(self):
        """
        Localization position within ROI.
        """
        lpos = self.pos*1
        lpos[:,0] -= self.data.roipos[:,-1]
        lpos[:,1] -= self.data.roipos[:,-2]
        return lpos
        
    def __repr__(self):
        return f'Dataset with {len(self)} {self.dims}D localizations ({self.imgshape[1]}x{self.imgshape[0]} pixel image).'
    
    def FRC2D(self, zoom=10, display=True,pixelsize=None):
        if pixelsize is None:
            pixelsize = self['pixelsize']
        return FRC(self.pos[:,:2], self.photons, zoom, self.imgshape, pixelsize, display=display)

    def estimateDriftMinEntropy(self, framesPerBin=10,maxdrift=5,initialEstimate=None,  apply=False, dims=None, pixelsize=None, **kwargs):
        if dims is None:
            dims = self.dims

        if pixelsize is None:
            pixelsize = self['pixelsize']

        drift, _, est_precision = minEntropy(self.data.estim.pos[:,:dims], 
                   self.data.frame,
                   self.data.crlb.pos[:,:dims], framesperbin=framesPerBin, 
                   maxdrift=maxdrift,initialEstimate=initialEstimate,
                   imgshape=self.imgshape, pixelsize=pixelsize, **kwargs)
        
        if apply:
            self.applyDrift(drift)

        return drift, est_precision
        
    def applyDrift(self, drift):
        if drift.shape[1] != self.dims:
            print(f"Applying {drift.shape[1]}D drift to {self.dims}D localization dataset")
        self.data.estim.pos[:,:drift.shape[1]] -= drift[self.data.frame]
        self.config['drift'] = drift
        
    @property
    def isDriftCorrected(self):
        return 'drift' in self.config
        
    def undoDrift(self):
        if not self.isDriftCorrected:
            raise ValueError('No drift correction has been done on this dataset')
        
        drift = self['drift']
        self.data.estim.pos[:,:drift.shape[1]] += drift[self.frame]
        
    def _xyI(ds):
        r=np.zeros((len(ds),3))
        r[:,:2] = ds.pos[:,:2]
        r[:,2] = ds.photons
        return r

    
    def estimateDriftRCC(self, framesPerBin=500, zoom=1, maxdrift=3):
        drift = rcc(self._xyI(), self.framenum, int(self.numFrames/framesPerBin), 
            np.max(self.imgshape), maxdrift=maxdrift, zoom=zoom)[0]
        return drift
        
    def estimateDriftFiducialMarkers(self, marker_indices, median_filter_size=None):
        """
        Marker indices is a list of lists with indices
        """

        drift = np.zeros((self.numFrames, self.dims))
        for marker_idx in marker_indices:
            t = (self.data.frame[marker_idx]+0.5)
            sortedidx = np.argsort(t)
            # Throw out edge cases where two localizations of a bead have the same frame. 
            # This should never happen of course so actually something else is messed up (bad beads, wrong thresholds)
            sortedidx = sortedidx[:-1][np.diff(t[sortedidx])>0]
            
            for d in range(self.dims):
                spl = InterpolatedUnivariateSpline(t[sortedidx], self.pos[marker_idx[sortedidx]][:,d], k=1)
                trace = spl(np.arange(self.numFrames))
                drift[:,d] += trace - np.mean(trace)
        drift /= len(marker_indices)
        
        unfiltered = drift*1
        
        if median_filter_size is not None:
            for i in range(self.dims):
                drift[:,i] = median_filter(drift[:,i], size=median_filter_size, mode='nearest')

        unfiltered-=np.mean(unfiltered,0)
        drift-=np.mean(drift,0)
        return drift,unfiltered
    
        
    def align(self, other):
        xyI = np.concatenate([self._xyI(), other._xyI()])
        framenum = np.concatenate([np.zeros(len(self),dtype=np.int32), np.ones(len(other),dtype=np.int32)])
        
        return 2*rcc(xyI, framenum, 2, np.max(self.imgshape), maxdrift=10,zoom=2,RCC=False)[0][1]

    @property
    def fields(self):
        return self.data.dtype.fields

    
    def renderGaussianSpots(self, zoom, sigma=1):
        imgshape = np.array(self.imgshape)*zoom
        with Context() as ctx:

            img = np.zeros(imgshape,dtype=np.float64)
            if sigma is None:
                sigma = np.mean(self.crlb.pos[:2])
            
            spots = np.zeros((len(self), 5), dtype=np.float32)
            spots[:, 0] = self.pos[:,0] * zoom
            spots[:, 1] = self.pos[:,1] * zoom
            spots[:, 2] = sigma
            spots[:, 3] = sigma
            spots[:, 4] = self.photons

            return GaussianPSFMethods(ctx).Draw(img, spots)
        
    
    def pick(self, centers, distance, debugMode=False):
        with Context(debugMode=debugMode) as ctx:
            counts, indices = PostProcessMethods(ctx).FindNeighbors(centers, self.pos, distance)

        idxlist = []
        pos = 0
        for count in counts:
            idxlist.append( indices[pos:pos+count] )
            pos += count
            
        return idxlist
        

    def cluster(self, maxDistance, debugMode=False):
        with Context(debugMode=debugMode) as ctx:
                        
            def callback(startidx, counts, indices):
                print(f"Callback: {startidx}. counts:{len(counts)} indices:{len(indices)}")
                                                        
            clusterPos, clusterCrlb, mapping = PostProcessMethods(ctx).ClusterLocs(
                self.pos, self.crlb.pos, maxDistance)
                    
        print(f"Computing cluster properties")
        
        counts = np.bincount(mapping)
        def getClusterData(org):
            r = np.recarray( len(counts), dtype=self.dtypeEstim)
            r.photons = np.bincount(mapping, org.photons) / counts
            r.bg = np.bincount(mapping, org.bg) / counts
            for k in range(self.dims):
                r.pos[:,k] = np.bincount(mapping, org.pos[:,k]) / counts
            return r
                
        clusterEstim = getClusterData(self.data.estim)
        clusterCrlb = getClusterData(self.data.crlb)
        
        ds = Dataset(len(clusterPos), self.dims, self.imgshape, config=self.config)
        ds.data.estim = clusterEstim
        ds.data.crlb = clusterCrlb
        ds.sigma = np.ones(self.dims)*maxDistance
        
        clusters = [[] for i in range(len(ds))]
        for i in range(len(mapping)):
            clusters[mapping[i]].append(i)
        for i in range(len(clusters)):
            clusters[i] = np.array(clusters[i])
        
        return ds, mapping, clusters#clusterPos, clusterCrlb, mapping 
    
    def scale(self, s):
        s=np.array(s)
        self.pos *= s[None,:]
        self.crlb.pos *= s[None,:]
    
    def save(self, fn, **kwargs):
        ext = os.path.splitext(fn)[1]
        if ext == '.hdf5':
            return self.saveHDF5(fn, **kwargs)
        elif ext == '.3dlp':
            return self.saveVisp3DLP(fn)
        elif ext == '.csv': # thunderstorm compatible csv
            return self.saveCSVThunderStorm(fn)
        else:
            raise ValueError('unknown extension')
            
    def saveVisp3DLP(self,fn):
        # x,y,z,lx,ly,lz,i,f
        data = np.zeros((len(self),8),dtype=np.float32)
        data[:,:3] = self.pos
        data[:,3:6] = self.crlb.pos
        data[:,6] = self.photons
        data[:,7] = self.frame
        
        np.savetxt(fn, data, fmt='%.3f %.3f %.3f %.3f %.3f %.3f %.0f %d')
        

    def saveHDF5(self, fn, saveGroups=False):
        print(f"Saving hdf5 to {fn}")
    
        with h5py.File(fn, 'w') as f:
            dtype = [('frame', '<u4'), 
                     ('x', '<f4'), ('y', '<f4'),
                     ('photons', '<f4'), 
                     ('sx', '<f4'), ('sy', '<f4'), 
                     ('bg', '<f4'), 
                     ('lpx', '<f4'), ('lpy', '<f4'), 
                     ('lI', '<f4'), ('lbg', '<f4'), 
                     ('ellipticity', '<f4'), 
                     ('net_gradient', '<f4'),
                     ('roi_index', '<i4'),
                     ('chisq', '<f4')]
            
            if saveGroups:
                dtype.append(('group', '<u4'))

            if 'sigma' in self.data.estim.dtype.fields:
                dtype.append(('lsx', '<f4'))
                dtype.append(('lsy', '<f4'))
            
            if self.dims==3:
                for fld in [('z', '<f4'), ('lpz', '<f4')]:
                    dtype.append(fld)
            
            locs = f.create_dataset('locs', shape=(len(self),), dtype=dtype)
            locs['frame'] = self.frame
            locs['x'] = self.pos[:,0]
            locs['y'] = self.pos[:,1]
            locs['lpx'] = self.crlb.pos[:,0]
            locs['lpy'] = self.crlb.pos[:,1]

            if self.dims==3:
                locs['z'] = self.pos[:,2]
                locs['lpz'] = self.crlb.pos[:,2]
                        
            locs['photons'] = self.photons
            locs['bg'] = self.background
            if 'sigma' in self.data.estim.dtype.fields:
                locs['sx'] = self.data.estim.sigma[:,0]
                locs['sy'] = self.data.estim.sigma[:,1]
                locs['lsx'] = self.crlb.sigma[:,0]
                locs['lsy'] = self.crlb.sigma[:,1]
            locs['lI'] = self.crlb.photons,
            locs['lbg'] = self.crlb.bg
            locs['net_gradient'] = 0
            locs['chisq'] = self.chisq
            locs['roi_index'] = self.data.roi_id # index into original un-filtered list of detected ROIs
            
            if saveGroups:
                locs['group'] = self.data.group
                                
            info =  {'Byte Order': '<',
                     'Camera': 'Dont know' ,
                     'Data Type': 'uint16',
                     'File': fn,
                     'Frames': int(np.max(self.frame)+1 if len(self.frame)>0 else 0),
                     'Width': int(self.imgshape[1]),
                     'Height': int(self.imgshape[0])
                     }
            
            info_fn = os.path.splitext(fn)[0] + ".yaml" 
            with open(info_fn, "w") as file:
                yaml.dump(info, file, default_flow_style=False) 
                
    def saveCSVThunderStorm(self, fn):
        """
        Save thunderstorm compatible CSV file
        """
        #"frame","x [nm]","y [nm]","sigma [nm]","intensity [photon]","offset [photon]","bkgstd [photon]","chi2","uncertainty [nm]"
        fields = ['frame', 'x', 'y', 'sigma', 'intensity', 'offset', 'uncertainty', 'crlbx', 'crlby']
        if 'extraFields' in self.config:
            for f in self.config['extraFields']:
                fields.append(f[0])
            
        data = np.zeros((len(self),len(fields)))
        data[:,0] = self.frame
        data[:,1:3] = self.pos[:,:2]
        data[:,3] = np.mean(self.data.estim.sigma,1)
        data[:,4] = self.photons
        data[:,5] = self.background
        data[:,6] = np.mean(self.crlb.pos[:,:2],1)
        data[:,7] = self.crlb.pos[:,0]
        data[:,8] = self.crlb.pos[:,1]
        if 'extraFields' in self.config:
            for i,f in enumerate(self.config['extraFields']):
                data[:,7+i] = self.data[f[0]]
        
        header= ','.join([f'"{v}"' for v in fields])
        np.savetxt(fn, data, fmt='%.6f', delimiter=',', header=header, comments='')
    
    def autocrop(self):
        minpos = np.min(self.pos[:,:2],0)
        maxpos = np.max(self.pos[:,:2],0)
        print(f"Min: {minpos}, max: {maxpos}")
        return self.crop(minpos,maxpos)
    
    def crop(self, minpos, maxpos, silent=False):
        minpos = np.array(minpos)
        maxpos = np.array(maxpos)
        which = (self.pos >= minpos[None]) & (self.pos <= maxpos[None])
        which = np.all(which,1)
        ds = self[which]
        ds.imgshape = np.ceil((maxpos-minpos)[[1,0]]).astype(int) # imgshape has array index order instead of x,y,z
        ds.pos -= minpos[None]
        if not silent:
            print(f"Cropping dataset. New shape: {ds.imgshape}, keeping {np.sum(which)}/{len(self)} spots")
        
        return ds
    
    def distanceToBorder(self):
        
        dist = self.pos[:,0]
        dist = np.minimum(dist, self.imgshape[1]-1-self.pos[:,0])
        dist = np.minimum(dist, self.pos[:,1])
        dist = np.minimum(dist, self.imgshape[0]-1-self.pos[:,1])
        
        return dist
    
    @staticmethod
    def loadHDF5(fn, **kwargs):
        
        with h5py.File(fn, 'r') as f:
            locs = f['locs'][:]
                        
            info_fn = os.path.splitext(fn)[0] + ".yaml" 
            with open(info_fn, "r") as file:
                if hasattr(yaml, 'unsafe_load'):
                    obj = yaml.unsafe_load_all(file)
                else:
                    obj = yaml.load_all(file)
                obj=list(obj)[0]
                imgshape=np.array([obj['Height'],obj['Width']])

            if 'z' in locs.dtype.fields:
                dims = 3
            else:
                dims = 2

            ds = Dataset(len(locs), dims, imgshape, haveSigma = 'sx' in locs.dtype.fields, **kwargs)
            ds.photons[:] = locs['photons']
            ds.background[:] = locs['bg']
            ds.pos[:,0] = locs['x']
            ds.pos[:,1] = locs['y']
            if dims==3: 
                ds.pos[:,2] = locs['z']
                ds.crlb.pos[:,2] = locs['lpz']

            ds.crlb.pos[:,0] = locs['lpx']
            ds.crlb.pos[:,1] = locs['lpy']
            ds.crlb.photons = locs['lI']
            ds.crlb.bg = locs['lbg']

            if 'lsx' in locs.dtype.fields: # picasso doesnt save crlb for the sigma fits
                ds.crlb.sigma[:,0] = locs['lsx']
                ds.crlb.sigma[:,1] = locs['lsy']
            
            if ds.hasPerSpotSigma():
                ds.data.estim.sigma[:,0] = locs['sx']
                ds.data.estim.sigma[:,1] = locs['sy']
            
            ds.frame[:] = locs['frame']
            
            if 'chisq' in locs.dtype.fields:
                ds.data.chisq = locs['chisq']
            
            if 'group' in locs.dtype.fields:
                ds.data.group = locs['group']
            
        ds['locs_path'] = fn
        return ds
    
    @staticmethod
    def loadCSVThunderStorm(fn, **kwargs):
        """
        Load CSV, thunderstorm compatible
        """

        data = np.genfromtxt(fn, delimiter=',',skip_header=0,names=True)
        dims = 3 if 'z_nm' in data.dtype.fields else 2
        
        imgshape = [
            int(np.ceil(np.max(data['y_nm']))), 
            int(np.ceil(np.max(data['x_nm'])))
        ]
        
        usedFields = ['x_nm', 'y_nm', 'z_nm', 'intensity_photon', 'offset_photon', 'chi2', 'uncertainty_nm', 'sigma_nm', 'frame']
        extraFields = []
        for f in data.dtype.fields:
            if not f in usedFields:
                extraFields.append((f, data.dtype.fields[f][0]))

        ds = Dataset(len(data), dims, imgshape, pixelsize=1, haveSigma='sigma_nm' in data.dtype.fields, extraFields=extraFields)
        ds.pos[:,0] = data['x_nm']
        ds.pos[:,1] = data['y_nm']
        if dims==3: ds.pos[:,2] = data['z_nm']
        
        ds.data.estim.sigma = data['sigma_nm'][:,None]
        ds.data.frame = data['frame'].astype(np.int32)
        ds.data.crlb.pos[:,:2] = data['uncertainty_nm'][:,None]
        
        ds.photons[:] = data['intensity_photon']
        ds.background[:] = data['offset_photon']
        ds.chisq[:] = data['chi2']
        
        for f in data.dtype.fields:
            if not f in usedFields:
                ds.data[f] = data[f]

        ds['locs_path'] = fn
                
        return ds
        
        
    @staticmethod
    def load(fn, **kwargs):
        ext = os.path.splitext(fn)[1]
        if ext == '.hdf5':
            return Dataset.loadHDF5(fn, **kwargs)
        elif ext == '.csv':
            return Dataset.loadCSVThunderStorm(fn, **kwargs)
        else:
            raise ValueError('unknown extension')
    
    @staticmethod
    def fromEstimates(estim, param_names, framenum, imgshape, crlb=None, chisq=None, roipos=None, addroipos=True, **kwargs):
        
        is3D = 'z' in param_names
        haveSigma = 'sx' in param_names
        if haveSigma:
            sx = param_names.index('sx')
            sy = param_names.index('sy')
        else:
            sx=sy=None
            
        dims = 3 if is3D else 2
        I_idx = param_names.index('I')
        bg_idx = param_names.index('bg')
        
        ds = Dataset(len(estim), dims, imgshape, haveSigma=haveSigma, **kwargs)
        ds.data.roi_id = np.arange(len(estim))
        
        if estim is not None:
            if addroipos and roipos is not None:
                estim = estim*1
                estim[:,[0,1]] += roipos[:,[1,0]]

            if np.can_cast(estim.dtype, ds.dtypeEstim):
                ds.data.estim = estim
            else:
                # Assuming X,Y,[Z,]I,bg
                ds.data.estim.pos = estim[:,:dims]
                ds.data.estim.photons = estim[:,I_idx]
                ds.data.estim.bg = estim[:,bg_idx]
                
                if haveSigma:
                    ds.data.estim.sigma = estim[:,[sx,sy]]
                    ds.sigma = np.median(ds.data.estim.sigma,0)
            
        if crlb is not None:
            if np.can_cast(crlb.dtype, ds.dtypeEstim):
                ds.data.crlb = crlb
            else:
                ds.data.crlb.pos = crlb[:,:dims]
                ds.data.crlb.photons = crlb[:,I_idx]
                ds.data.crlb.bg = crlb[:,bg_idx]

                if haveSigma:
                    ds.data.crlb.sigma = crlb[:,[sx,sy]]
            
        if chisq is not None:
            ds.data.chisq = chisq
        
        if framenum is not None:
            ds.data.frame = framenum
            
        if roipos is not None:
            ds.data.roipos = roipos
            
        return ds
    
    def info(self):
        m_crlb_x = np.median(self.crlb.pos[:,0])
        m_bg= np.median(self.background)
        m_I=np.median(self.photons)
        return f"#Spots: {len(self)}. Imgsize:{self.imgshape[0]}x{self.imgshape[1]} pixels. Median CRLB X: {m_crlb_x:.2f} [pixels], bg:{m_bg:.1f}. I:{m_I:.1f}"
    
    def simulateBlinking(self, numframes, blinkGenerator, drift=None):
        data_per_frame = []

        for f,ontimes in enumerate(tqdm.tqdm(blinkGenerator, total=numframes, desc='Simulating blinking molecules')):
            idx = np.where(ontimes>0)[0]

            d = np.recarray(len(idx), dtype=self.data.dtype)
            d.fill(0)

            d[:] = self.data[idx]
            d.frame = f

            if drift is not None:
                d.estim.pos[:,:self.dims] += drift[f,None]

            d.estim.photons = self.photons[idx] * ontimes[idx]
            data_per_frame.append(d)
            
        data = np.concatenate(data_per_frame).view(np.recarray)
        ds = Dataset(0, self.dims, self.imgshape, data, config=self.config)
        return ds
            
    
    def simulateMovie(self, psf : Estimator, tiff_fn, background=5, drift=None, returnMovie=False):
        if len(psf.sampleshape) != 2:
            raise ValueError('Expecting a PSF with 2D image output')
            
        with MultipartTiffSaver(tiff_fn) as tifwriter:
            
            ipf = self.indicesPerFrame()
            
            if returnMovie:
                movie = np.zeros((self.numFrames, *self.imgshape),dtype=np.uint16)

            for f,idx in enumerate(tqdm.tqdm(ipf, total=len(ipf), desc=f'Rendering movie to {tiff_fn}')):
                params = np.zeros((len(idx),2+self.dims))
                params[:,:self.dims] = self.pos[idx]
                if drift is not None:
                    params[:,:self.dims] += drift[f]
                params[:,self.dims] = self.photons[idx]
                roisize = np.array(psf.sampleshape)
                roipos = (params[:,[1,0]] - roisize//2).astype(np.int32)
                params[:,[1,0]] -= roipos
                
                rois = psf.ExpectedValue(params)
                img = psf.ctx.smlm.DrawROIs(self.imgshape, rois, roipos)
                img += background
                smp = np.random.poisson(img)
                tifwriter.save(smp)

                if returnMovie:
                    movie[f] = smp

        if returnMovie:
            return movie
    
    @staticmethod
    def fromQueueResults(qr, imgshape, **kwargs) -> 'Dataset':
        return Dataset.fromEstimates(qr.estim,  qr.param_names, qr.ids, imgshape, qr.crlb, qr.chisq, roipos=qr.roipos, **kwargs)
    