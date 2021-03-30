"""
Main SIMFLUX data processing pipeline.

photonpy - Single molecule localization microscopy library
© Jelmer Cnossen 2018-2021
"""
import numpy as np
import matplotlib.pyplot as plt
from photonpy.simflux.spotlist import SpotList
import math
import os,pickle
from photonpy import Context, Dataset
import sys
import time
import tqdm
from photonpy.cpp.estimator import Estimator
from photonpy.cpp.estim_queue import EstimQueue,EstimQueue_Results
from photonpy.cpp.roi_queue import ROIQueue
from photonpy.cpp.gaussian import Gaussian
import photonpy.cpp.spotdetect as spotdetect
from photonpy.cpp.simflux import SIMFLUX, CFSFEstimator 
from scipy.interpolate import InterpolatedUnivariateSpline

import photonpy.utils.multipart_tiff as read_tiff
import photonpy.smlm.process_movie as process_movie
from photonpy.smlm.util import plot_traces
from photonpy.smlm import blinking_spots
from photonpy.cpp.postprocess import PostProcessMethods
from photonpy.utils import multipart_tiff


figsize=(9,7)

ModDType = SIMFLUX.modulationDType

#mpl.use('svg')

# Make sure the angles dont wrap around, so you can plot them and take mean
# TODO: loop through this..
def unwrap_angle(ang):
    r = ang * 1
    ang0 = ang.flatten()[0]
    r[ang > ang0 + math.pi] -= 2 * math.pi
    r[ang < ang0 - math.pi] += 2 * math.pi
    return r


# Pattern angles wrap at 180 degrees
def unwrap_pattern_angle(ang):
    r = ang * 1
    ang0 = ang.flatten()[0]
    r[ang > ang0 + math.pi / 2] -= math.pi
    r[ang < ang0 - math.pi / 2] += math.pi
    return r


def print_phase_info(mod):
    for axis in [0, 1]:
        steps = np.diff(mod[axis::2, 3])
        steps[steps > np.pi] = -2 * np.pi + steps[steps > np.pi]
        print(f"axis {axis} steps: {-steps*180/np.pi}")



def result_dir(path):
    dir, fn = os.path.split(path)
    return dir + "/results/" + os.path.splitext(fn)[0] + "/"


        
def load_mod(tiffpath):
    with open(os.path.splitext(tiffpath)[0]+"_mod.pickle", "rb") as pf:
        mod = pickle.load(pf)['mod']
        assert(mod.dtype == ModDType)
        return mod
    
    


def print_mod(reportfn, mod, pattern_frames, pixelsize):
    k = mod['k']
    phase = mod['phase']
    depth = mod['depth']
    ri = mod['relint']
    
    for i in range(len(mod)):
        reportfn(f"Pattern {i}: kx={k[i,0]:.4f} ky={k[i,1]:.4f} Phase {phase[i]*180/np.pi:8.2f} Depth={depth[i]:5.2f} "+
               f"Power={ri[i]:5.3f} ")

    for ang in range(len(pattern_frames)):
        pat=pattern_frames[ang]
        d = np.mean(depth[pat])
        phases = phase[pat]
        shifts = (np.diff(phases[-1::-1]) % (2*np.pi)) * 180/np.pi
        shifts[shifts > 180] = 360 - shifts[shifts>180]
        
        with np.printoptions(precision=3, suppress=True):
            reportfn(f"Angle {ang} shifts: {shifts} (deg) (patterns: {pat}). Depth={d:.3f}")
    
    
def equal_cache_cfg(output_fn, cfg, input_fn):
    """
    Returns true if the config file associated with data file data_fn contains the same value as cfg
    """ 
    mtime = os.path.getmtime(input_fn)
    cfg_fn = os.path.splitext(output_fn)[0]+"_cfg.pickle"
    if not os.path.exists(cfg_fn):
        return False
    with open(cfg_fn,"rb") as f:
        d = pickle.load(f)
        if len(d) != 2:
            return False
        
        stored, stored_mtime = d
        
        if stored_mtime != mtime:
            return False
        
        try:
            # Note that we can't just do 'return stored == cfg'. 
            # If one of the values in a dictionary is a numpy array, 
            # we will get "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()"
            # See https://stackoverflow.com/questions/26420911/comparing-two-dictionaries-with-numpy-matrices-as-values
            np.testing.assert_equal(stored, cfg)
        except:
            return False

        return True
            
def save_cache_cfg(output_fn, cfg, input_fn):
    mtime = os.path.getmtime(input_fn)
    cfg_fn = os.path.splitext(output_fn)[0]+"_cfg.pickle"
    with open(cfg_fn,"wb") as f:
        pickle.dump((cfg, mtime),f)
    
    
    
class SimfluxProcessor:
    """
    Simflux processing. 
    """
    def __init__(self, src_fn, cfg, debugMode=False):
        src_fn = os.path.abspath(src_fn)
        """
        chi-square threshold: Real threshold = chisq_threshold*roisize^2
        """
        self.pattern_frames = np.array(cfg['patternFrames'])
        self.src_fn = src_fn
        self.rois_fn = os.path.splitext(src_fn)[0] + "_rois.npy"
        self.debugMode = debugMode
        self.cfg = cfg
        self.sigma = cfg['sigma']
        if np.isscalar(self.sigma):
            self.sigma = [self.sigma, self.sigma]
        self.roisize = cfg['roisize']
        self.pixelsize = cfg['pixelsize']
        self.threshold = cfg['detectionThreshold']
        self.maxframes = cfg['maxframes'] if 'maxframes' in cfg else -1
        
        self.mod_fn = os.path.splitext(self.src_fn)[0]+"-mod.pickle"
        self.roi_indices=None
        
        self.IBg = None
        self.sum_fit = None
        
        dir, fn = os.path.split(src_fn)
        self.resultsdir = dir + "/results/" + os.path.splitext(fn)[0] + "/"
        os.makedirs(self.resultsdir, exist_ok=True)
        self.resultprefix = self.resultsdir
            
        self.reportfile = self.resultprefix + "report.txt"
        with open(self.reportfile,"w") as f:
            f.write("")
            
        self.g_undrifted=None
        self.sf_undrifted=None
        
    def _camera_calib(self, ctx):
        return process_movie.create_calib_obj(self.cfg['gain'], self.cfg['offset'],self.imgshape, ctx)

    def detect_rois(self, chisq_threshold=4, ignore_cache=False, roi_batch_size=20000, background_img=None):
        self.imgshape = read_tiff.tiff_get_image_size(self.src_fn)
        
        self.chisq_threshold = chisq_threshold

        spotDetector = spotdetect.SpotDetector(self.sigma, self.roisize, self.threshold, backgroundImage=background_img)
            
        if not equal_cache_cfg(self.rois_fn, self.cfg, self.src_fn) or ignore_cache:
            with Context(debugMode=self.debugMode) as ctx:
                process_movie.detect_spots(spotDetector, self._camera_calib(ctx), 
                                   read_tiff.tiff_read_file(self.src_fn, self.cfg['startframe'], self.maxframes), 
                                   self.pattern_frames.size, self.rois_fn, batch_size = roi_batch_size, ctx=ctx)
            save_cache_cfg(self.rois_fn, self.cfg, self.src_fn)
            
        self.numrois = int(np.sum([len(ri) for ri,px in self._load_rois_iterator()]))
        print(f"Num ROIs: {self.numrois}")
        
        if self.numrois == 0:
            raise ValueError('No spots detected')
        
        self.estimate_sigma()
        self.gaussian_fitting()

    def close(self):
        ...

    def estimate_sigma(self, initialSigma=2, plot=True):
        
        with Context(debugMode=self.debugMode) as ctx:
            pos_estim = Gaussian(ctx).CreatePSF_XYIBg(self.roisize, initialSigma, True)
            r = process_movie.localize_rois(self._summed_rois_iterator(), pos_estim, total=self.numrois)
            
            sigma_estim = Gaussian(ctx).CreatePSF_XYIBgSigmaXY(self.roisize, initialSigma, True)
            
            initial = np.zeros((self.numrois, 6))
            initial[:,:4] = r.estim
            initial[:,4:] = initialSigma
            
            r = process_movie.localize_rois(self._summed_rois_iterator(), sigma_estim, initial_estim=initial)
        
        if plot:
            plt.figure()
            plt.hist(r.estim [:,4], range=[1, 4],bins=200)
            plt.title('Sigma X')
        
            plt.figure()
            plt.hist(r.estim[:,5], range=[1, 4],bins=200)
            plt.title('Sigma Y')
        
        best = np.median(r.estim [:,[4,5]],0)
        print(f'Now using estimated sigma: {best}')
        
        self.cfg['sigma'] = best
        self.sigma = best
        return best
        
    def view_rois(self, indices=None, summed=False, fits=None):
        import napari
        
        ri, pixels = process_movie.load_rois(self.rois_fn)
        
        if self.roi_indices is not None:
            px = pixels[self.roi_indices]
        else:
            px = pixels

        if indices is not None:
            px = px[indices]
        
        if summed:
            px = px.sum(1)
        
        with napari.gui_qt():
            viewer = napari.view_image(px)

            if fits is not None:
                #points = np.array([[100, 100], [200, 200], [300, 100]])
                
                for data, kwargs in fits:
                    coords = np.zeros((len(data),3))
                    coords[:,0] = np.arange(len(data))
                    coords[:,[2,1]] = data[:,:2]
      
                    viewer.add_points(coords, size=0.1, **kwargs)
                
        return viewer
    
    def gaussian_fitting(self):
        """
        Make sure self.IBg and self.sum_fits are known
        """
        if self.IBg is not None and self.sum_fit is not None:
            return
        
        rois_info = []
        sum_fit = []
        ibg = []
        sum_crlb = []
        sum_chisq = []
        
        sigma = np.array(self.sigma)
        
        print('2D Gaussian fitting...',flush=True)
        
        with Context(debugMode=self.debugMode) as ctx:
            gaussfn = Gaussian(ctx)
            with gaussfn.CreatePSF_XYIBg(self.roisize, self.sigma, True) as psf, tqdm.tqdm(total=self.numrois) as pb:
                for ri, pixels in self._load_rois_iterator():
                    summed = pixels.sum(1)
                    e = psf.Estimate(summed)[0]
                    sum_crlb.append(psf.CRLB(e))
                    sum_chisq.append(psf.ChiSquare(e, summed))
                    
                    rois_info.append(ri)
                    
                    sh = pixels.shape # numspots, numpatterns, roisize, roisize
                    pixels_rs = pixels.reshape((sh[0]*sh[1],sh[2],sh[3]))
                    xy = np.repeat(e[:,[0,1]], sh[1], axis=0)
                    
                    ibg_, crlb_ = gaussfn.EstimateIBg(pixels_rs, sigma[None], xy,useCuda=True)
                    ic = np.zeros((len(e)*sh[1],4))
                    ic [:,[0,1]] = ibg_
                    ic [:,[2,3]] = crlb_
                    ibg.append(ic.reshape((sh[0],sh[1],4)))
                                    
                    sum_fit.append(e)
    
                    pb.update(len(pixels))
                    
                param_names = psf.param_names
        print(flush=True)

        sum_fit = np.concatenate(sum_fit)
        IBg = np.concatenate(ibg)
        sum_chisq = np.concatenate(sum_chisq)
        sum_crlb = np.concatenate(sum_crlb)
        rois_info = np.concatenate(rois_info)

        self._store_IBg_fits(sum_fit, IBg, sum_chisq, sum_crlb, rois_info, param_names)

    def _store_IBg_fits(self, sum_fit, ibg, sum_chisq, sum_crlb, rois_info,psf_param_names,fov=None, roi_indices=None):
    
        self.sum_fit = sum_fit
        self.IBg = ibg
        self.sum_chisq = sum_chisq
        self.sum_crlb = sum_crlb

        roipos = np.zeros((len(rois_info),3), dtype=np.int32)
        roipos[:,0] = 0
        roipos[:,1] = rois_info['y']
        roipos[:,2] = rois_info['x']
        self.roipos = roipos
        
        plt.figure()
        plt.hist(self.sum_chisq, bins=50, range=[0,4000])
        plt.title('Non-simflux fit chi-square')
        
        if self.chisq_threshold>0:
            threshold = self.roisize**2 * self.chisq_threshold
            plt.axvline(threshold, label='Threshold')
            ok = self.sum_chisq < threshold
            print(f"Accepted {np.sum(ok)}/{self.numrois} spots (chi-square threshold={threshold:.1f}")
        else:
            ok = np.ones(self.sum_chisq.shape, dtype=np.bool)
        
        border = self.roisize/4
        ok = ok & ( 
            (self.sum_fit[:,0] > border) & 
            (self.sum_fit[:,1] > border) & 
            (self.sum_fit[:,0] < self.roisize-border-1) &
            (self.sum_fit[:,1] < self.roisize-border-1))
        

        self.roipos = self.roipos[ok]
        self.sum_fit = self.sum_fit[ok]
        self.IBg = self.IBg[ok]
        self.sum_chisq = self.sum_chisq[ok]
        self.sum_crlb = self.sum_crlb[ok]
        self.framenum = rois_info['id'][ok]
        
        if roi_indices is not None:
            self.roi_indices = roi_indices[ok]
        else:
            self.roi_indices = np.where(ok)[0]
        
        self.sum_ds = Dataset.fromEstimates(self.sum_fit, psf_param_names, self.framenum, 
                              self.imgshape, crlb=self.sum_crlb, chisq=self.sum_chisq, 
                              roipos=self.roipos[:,1:])
        
        fn = self.resultprefix+'g2d_fits.hdf5'
        self.sum_ds.save(fn)

        self.spotlist = SpotList(self.sum_ds, self.selected_roi_source, pixelsize=self.cfg['pixelsize'], 
                            outdir=self.resultsdir, IBg=self.IBg[:,:,:2], IBg_crlb=self.IBg[:,:,2:])
        
        median_crlb_x = np.median(self.sum_ds.crlb.pos[:,0])
        median_I = np.median(self.sum_ds.photons)

        self.report(f"g2d mean I={median_I:.1f}. mean crlb x {median_crlb_x:.4f}")
        
    def save_pickle(self, fn):
        with open(fn, "wb") as f:
            pickle.dump((
                self.sum_fit, 
                self.IBg,
                self.sum_crlb,
                self.sum_chisq,
                self.roipos,
                self.framenum,
                self.sum_ds), f)

    def load_pickle(self, fn):
        
        with open(fn, "rb") as f:
            self.sum_fit, self.IBg, self.sum_crlb, self.sum_chisq, self.roipos,  \
                self.framenum, self.sum_ds = pickle.load(f)
        
        self.spotlist = SpotList(self.sum_ds, self.selected_roi_source, pixelsize=self.cfg['pixelsize'], 
                    outdir=self.resultsdir, IBg=self.IBg[:,:,:2], IBg_crlb=self.IBg[:,:,2:])

                        
    def estimate_patterns(self, num_angle_bins=1,
                          num_phase_bins=10, 
                          pitch_minmax_nm=[200,300], 
                          fix_phase_shifts=None, 
                          fix_depths=None,
                          show_plots=True,
                          debug_images=False,
                          dft_peak_search_range=0.02,
                          phase_method=1):
        
        freq_minmax = 2*np.pi / (np.array(pitch_minmax_nm[::-1]) / self.pixelsize)
        
        nframes = self.sum_ds.numFrames
        fr = np.arange(nframes)
    
        with Context(debugMode=self.debugMode) as ctx:
            angles, pitch = self.spotlist.estimate_angle_and_pitch(
                self.pattern_frames, 
                frame_bins=np.array_split(fr, num_angle_bins), 
                ctx=ctx,
                freq_minmax=freq_minmax,
                debug_images=debug_images,
                dft_peak_search_range=dft_peak_search_range
            )
        
        num_patterns = self.pattern_frames.size
        mod = np.zeros((num_patterns),dtype=ModDType)

        print("Pitch and angle estimation: ")
        for k in range(len(self.pattern_frames)):
            angles[angles[:, k] > 0.6 * np.pi] -= np.pi  # 180 deg to around 0
            angles[:, k] = unwrap_pattern_angle(angles[:, k])
            angles_k = angles[:, k]
            pitch_k = pitch[:, k]
            self.report(f"Angle {k}: { np.rad2deg(np.mean(angles_k)) :7.5f} [deg]. Pitch: {np.mean(pitch_k)*self.pixelsize:10.5f} ({2*np.pi/np.mean(pitch_k):3.3f} [rad/pixel])")

            freq = 2 * np.pi / np.mean(pitch_k)
            kx = np.cos(np.mean(angles_k)) * freq
            ky = np.sin(np.mean(angles_k)) * freq
            mod['k'][self.pattern_frames[k], :2] = kx,ky
            
        if 'use3D' in self.cfg and self.cfg['use3D']:
            wavelen = self.cfg['wavelength']
            k = mod['k'] / self.cfg['pixelsize'] # k is now in rad/nm
            kr = np.sqrt( (k[:,:2]**2).sum(-1) )
            beamAngle = np.arcsin(kr * wavelen  / (4*np.pi)) * 2
            print(f"Beam angles w.r.t. optical axis: {np.rad2deg(beamAngle)}")
            kz = 2*np.pi/wavelen *np.cos(beamAngle) * 1000 # rad/um
            
            mod['k'][:,2] = 0#-kz*0.25
            self.beamAngle = beamAngle
            self.excDims = 3
        else:
            self.excDims = 2
                        
        frame_bins = np.array_split(fr, num_phase_bins)
        frame_bins = [b for b in frame_bins if len(b)>0]
        
        k = mod['k'][:,:self.excDims]
        phase, depth, power = self.spotlist.estimate_phase_and_depth(k, self.pattern_frames, frame_bins, method=phase_method)
        phase_all, depth_all, power_all = self.spotlist.estimate_phase_and_depth(k, self.pattern_frames, [fr], method=phase_method)

        # store interpolated phase for every frame
        frame_bin_t = [np.mean(b) for b in frame_bins]
        self.phase_interp = np.zeros((nframes,num_patterns))
        for k in range(num_patterns):
            phase[:,k] = unwrap_angle(phase[:, k])
            spl = InterpolatedUnivariateSpline(frame_bin_t, phase[:,k], k=2)
            self.phase_interp[:,k] = spl(fr)
            
        fig = plt.figure(figsize=figsize)
        styles = ['o', "x", "*", 'd']
        for ax, idx in enumerate(self.pattern_frames):
            for k in range(len(idx)):
                p=plt.plot(fr, self.phase_interp[:,idx[k]] * 180/np.pi,ls='-')
                plt.plot(frame_bin_t, phase[:,idx[k]] * 180 / np.pi,ls='', c=p[0].get_color(), marker=styles[ax%len(styles)], label=f"Phase {idx[k]} (axis {ax})")
        plt.legend()
        plt.title(f"Phases for {self.src_fn}")
        plt.xlabel("Frame number"); plt.ylabel("Phase [deg]")
        plt.grid()
        plt.tight_layout()
        fig.savefig(self.resultprefix + "phases.png")
        if not show_plots: plt.close(fig)

        fig = plt.figure(figsize=figsize)
        for ax, idx in enumerate(self.pattern_frames):
            for k in range(len(idx)):
                plt.plot(frame_bin_t, depth[:, idx[k]], styles[ax%len(styles)], ls='-', label=f"Depth {idx[k]} (axis {ax})")
        plt.legend()
        plt.title(f"Depths for {self.src_fn}")
        plt.xlabel("Frame number"); plt.ylabel("Modulation Depth")
        plt.grid()
        plt.tight_layout()
        fig.savefig(self.resultprefix + "depths.png")
        if not show_plots: plt.close(fig)

        fig = plt.figure(figsize=figsize)
        for ax, idx in enumerate(self.pattern_frames):
            for k in range(len(idx)):
                plt.plot(frame_bin_t, power[:, idx[k]], styles[ax%len(styles)], ls='-', label=f"Power {idx[k]} (axis {ax})")
        plt.legend()
        plt.title(f"Power for {self.src_fn}")
        plt.xlabel("Frame number"); plt.ylabel("Modulation Power")
        plt.grid()
        plt.tight_layout()
        fig.savefig(self.resultprefix + "power.png")
        if not show_plots: plt.close(fig)

        # Update mod
        phase_std = np.zeros(len(mod))
        for k in range(len(mod)):
            ph_k = unwrap_angle(phase[:, k])
            mod['phase'][k] = phase_all[0, k]
            mod['depth'][k] = depth_all[0, k]
            mod['relint'][k] = power_all[0, k]
            phase_std[k] = np.std(ph_k)

        s=np.sqrt(num_phase_bins)
        for k in range(len(mod)):
            self.report(f"Pattern {k}: Phase {mod[k]['phase']*180/np.pi:8.2f} (std={phase_std[k]/s*180/np.pi:6.2f}) "+
                   f"Depth={mod[k]['depth']:5.2f} (std={np.std(depth[:,k])/s:5.3f}) "+
                   f"Power={mod[k]['relint']:5.3f} (std={np.std(power[:,k])/s:5.5f}) ")

        #mod=self.spotlist.refine_pitch(mod, self.ctx, self.spotfilter, plot=True)[2]

        if fix_phase_shifts:
            self.report(f'Fixing phase shifts to {fix_phase_shifts}' )
            phase_shift_rad = fix_phase_shifts / 180 * np.pi
            for ax in self.pattern_frames:
                mod[ax]['phase'] = mod[ax[0]]['phase'] + np.arange(len(ax)) * phase_shift_rad

            with Context(debugMode=self.debugMode) as ctx:
                mod=self.spotlist.refine_pitch(mod, self.ctx, self.spotfilter, plot=True)[2]

        for angIndex in range(len(self.pattern_frames)):
            mod[self.pattern_frames[angIndex]]['relint'] = np.mean(mod[self.pattern_frames[angIndex]]['relint'])
            # Average modulation depth
            mod[self.pattern_frames[angIndex]]['depth'] = np.mean(mod[self.pattern_frames[angIndex]]['depth'])

        mod['relint'] /= np.sum(mod['relint'])

        if fix_depths:
            self.report(f'Fixing modulation depth to {fix_depths}' )
            mod['depth']=fix_depths



        self.report("Final modulation pattern parameters:")
        print_mod(self.report, mod, self.pattern_frames, self.pixelsize)
        
        self.mod = mod
        with open(self.mod_fn,"wb") as f:
            pickle.dump((mod,self.phase_interp),f)
        
        med_sum_I = np.median(self.IBg[:,:,0].sum(1))
        lowest_power = np.min(self.mod['relint'])
        depth = self.mod[np.argmin(self.mod['relint'])]['depth']
        median_intensity_at_zero = med_sum_I * lowest_power * (1-depth)
        self.report(f"Median summed intensity: {med_sum_I:.1f}. Median intensity at pattern zero: {median_intensity_at_zero:.1f}")
        


    def pattern_plots(self, spotfilter=None):
        self.load_mod()
        self.report(f"Generating pattern plots using spot filter: {spotfilter}. ")
        for k in range(len(self.mod)):
            png_file= f"{self.resultprefix}patternspots{k}.png"
            print(f"Generating {png_file}...")
            src_name = os.path.split(self.src_fn)[1]
            self.spotlist.draw_spots_in_pattern(png_file, self.mod, 
                                       k, tiffname=src_name, numpts= 2000, spotfilter=spotfilter)
            self.spotlist.draw_spots_in_pattern(f"{self.resultprefix}patternspots{k}.svg", self.mod, 
                                       k, tiffname=src_name, numpts= 2000, spotfilter=spotfilter)

        self.spotlist.draw_axis_intensity_spread(self.pattern_frames, self.mod, spotfilter)
        
        
    def draw_mod(self, showPlot=False):
        allmod = self.mod
        filename = self.resultprefix+'patterns.png'
        fig,axes = plt.subplots(1,2)
        fig.set_size_inches(*figsize)
        for axis in range(len(self.pattern_frames)):
            axisname = ['X', 'Y']
            ax = axes[axis]
            indices = self.pattern_frames[axis]
            freq = np.sqrt(np.sum(allmod[indices[0]]['k']**2))
            period = 2*np.pi/freq
            x = np.linspace(0, period, 200)
            sum = x*0
            for i in indices:
                mod = allmod[i]
                q = (1+mod['depth']*np.sin(x*freq-mod['phase']) )*mod['relint']
                ax.plot(x, q, label=f"Pattern {i}")
                sum += q
            ax.plot(x, sum, label=f'Summed {axisname[axis]} patterns')
            ax.legend()
            ax.set_title(f'{axisname[axis]} modulation')
            ax.set_xlabel('Pixels');ax.set_ylabel('Modulation intensity')
        fig.suptitle('Modulation patterns')
        if filename is not None: fig.savefig(filename)
        if not showPlot: plt.close(fig)
        return fig
        
        
    def plot_ffts(self):
        with Context(debugMode=self.debugMode) as ctx:
            self.spotlist.generate_projections(self.mod, 4,ctx)
            self.spotlist.plot_proj_fft()
        
    def load_mod(self):
        with open(self.mod_fn, "rb") as f:
            self.mod, self.phase_interp = pickle.load(f)
            
    def set_mod(self, pitch_nm, angle_deg, depth, z_angle=None):
        """
        Assign mod array and phase_interp for simulation purposes
        """
        freq = 2*np.pi/np.array(pitch_nm)*self.pixelsize
        angle = np.deg2rad(angle_deg)
        self.mod = np.zeros(self.pattern_frames.size, dtype=ModDType)
        if z_angle is None:
            z_angle = angle*0
        else:
            z_angle = np.deg2rad(z_angle)
        for i,pf in enumerate(self.pattern_frames):
            self.mod['k'][pf,0] = np.cos(angle[i]) * freq[i] * np.cos(z_angle[i])
            self.mod['k'][pf,1] = np.sin(angle[i]) * freq[i] * np.cos(z_angle[i])
            self.mod['k'][pf,2] = freq[i] * np.sin(z_angle[i])
            self.mod['phase'][pf] = np.linspace(0,2*np.pi,len(pf),endpoint=False)

        self.mod['depth'] = depth
        self.mod['relint'] = 1/self.pattern_frames.size
        
        
    def mod_per_spot(self, framenums):
        """
        Return modulation patterns with spline interpolated phases
        """ 
        mod_ = np.tile(self.mod,len(framenums))
        
        for k in range(len(self.mod)):
            mod_['phase'][k::len(self.mod)] = self.phase_interp[framenums][:,k]
        
        return np.reshape(mod_.view(np.float32), (len(framenums), 6*len(self.mod)))
            
    def create_psf(self, ctx, modulated=False):
        if modulated:
            return SIMFLUX(ctx).CreateEstimator_Gauss2D(self.sigma,len(self.mod),
                                                        self.roisize,len(self.mod),
                                                        simfluxEstim=True)
        else:
            return Gaussian(ctx).CreatePSF_XYIBg(self.roisize, self.sigma, True)


    def process(self, spotfilter):
        self.load_mod()
                
        with Context(debugMode=self.debugMode) as ctx:
            moderrs = self.spotlist.compute_modulation_error(self.mod, spotfilter)
            self.report(f"RMS moderror: {np.sqrt(np.mean(moderrs**2)):.3f}")
        
            if len(self.pattern_frames)==2: # assume XY modulation
                self.draw_mod()
        
            #self.spotlist.bias_plot2D(self.mod, self.ctx, self.spotfilter, tag='')
        #        spotlist.plot_intensity_variations(mod, minfilter, pattern_frames)
        
            indices = self.spotlist.get_filtered_spots(spotfilter, self.mod)
        
            # g2d_results are the same set of spots used for silm, for fair comparison
            mod_ = self.mod_per_spot(self.sum_ds.frame)
            
            psf = self.create_psf(ctx, modulated=True)
            
            print(f"Constants per ROI: {psf.numconst}")

            with EstimQueue(psf, batchSize=2048, numStreams=2, keepSamples=False) as queue:

                idx = 0
                for roipos, pixels, block_indices in self.selected_roi_source(indices):
                    roi_mod = mod_[block_indices]
                    
                    queue.Schedule(pixels, roipos=roipos, 
                                   constants=roi_mod, ids=np.arange(len(roipos))+idx)
                    idx += len(roipos)
    
                queue.WaitUntilDone()
                qr = queue.GetResults()

            qr.SortByID(isUnique=True)
            qr.ids = self.sum_ds.frame[indices]
            border = 2.1
            idx = qr.FilterXY(border, border, self.roisize-border-1, self.roisize-border-1)
             
            # not sure yet if i need to allow dataset to have roipos with more than 2 dimensions (biplane?)
            qr.roipos = qr.roipos[:,1:]
            
            self.sf_ds = Dataset.fromQueueResults(qr, self.imgshape)

            self.result_indices = indices[idx]
            self.sum_ds_filtered = self.sum_ds[self.result_indices]
        
            self.sf_ds.save(self.resultprefix+"simflux.hdf5")
            self.sum_ds_filtered.save(self.resultprefix+"g2d-filtered.hdf5")
            

    
        
    def selected_roi_source(self, indices):
        """
        Yields roipos,pixels,idx for the selected ROIs. 
        'indices' indexes into the set of ROIs selected earlier by gaussian_fitting(), stored in roi_indices
        idx is the set of indices in the block, indexing into sum_fit
        """
        roi_idx = self.roi_indices[indices]
        
        mask = np.zeros(self.numrois, dtype=np.bool)
        mask[roi_idx] = True
        
        idx = 0
        
        indexmap = np.zeros(self.numrois,dtype=np.int32)
        indexmap[self.roi_indices[indices]] = indices
                
        for rois_info, pixels in process_movie.load_rois_iterator(self.rois_fn):
            block_mask = mask[idx:idx+len(pixels)] 
            block_roi_indices = indexmap[idx:idx+len(pixels)][block_mask]
            idx += len(pixels)
            
            if np.sum(block_mask) > 0:
                roipos = np.zeros((len(rois_info),3), dtype=np.int32)
                roipos[:,0] = 0
                roipos[:,1] = rois_info['y']
                roipos[:,2] = rois_info['x']
                
                yield roipos[block_mask], pixels[block_mask], block_roi_indices
                
        
    def crlb_map(self, intensity=None, bg=None, sample_area_width=0.2):
        """
        
        """
        if intensity is None:
            intensity = np.median(self.sum_ds.photons)
            
        if bg is None:
            bg = np.median(self.sum_ds.background)

        #pitchx = 2*np.pi / np.max(np.abs(self.mod['k'][:,0]))
        #pitchy = 2*np.pi / np.max(np.abs(self.mod['k'][:,1]))
                
        W = 100
        xr = np.linspace((0.5-sample_area_width/2)*self.roisize,(0.5+sample_area_width/2)*self.roisize,W)
        yr = np.linspace((0.5-sample_area_width/2)*self.roisize,(0.5+sample_area_width/2)*self.roisize,W)
        
        X,Y = np.meshgrid(xr,yr)
        
        with Context(debugMode=self.debugMode) as ctx:            
            sf_psf = self.create_psf(ctx, modulated=True)

            coords = np.zeros((W*W,sf_psf.numparams))
            coords[:,0] = X.flatten()
            coords[:,1] = Y.flatten()
            coords[:,-2] = intensity
            coords[:,-1] = bg 

            coords_ = coords*1
            coords_[:,-1] /= sf_psf.sampleshape[0] # bg should be distributed over all frames
            mod_ = np.repeat([self.mod.view(np.float32)], len(coords), 0)
            sf_crlb = sf_psf.CRLB(coords_, constants=mod_)
            
            psf = self.create_psf(ctx, modulated=False)
            g2d_crlb = psf.CRLB(coords)
        
        IFmap = g2d_crlb/sf_crlb
        
        fig,ax = plt.subplots(2,1,sharey=True)
        im = ax[0].imshow(IFmap[:,0].reshape((W,W)))
        ax[0].set_title('Improvement Factor X')

        ax[1].imshow(IFmap[:,1].reshape((W,W)))
        ax[1].set_title('Improvement Factor Y')

        fig.colorbar(im, ax=ax)
        
        IF = np.mean(g2d_crlb/sf_crlb,0)
        print(f"SF CRLB: {np.mean(sf_crlb,0)}")
        print(f"SMLM CRLB: {np.mean(g2d_crlb,0)}")
        
        print(f"Improvement factor X: {IF[0]:.3f}, Y: {IF[1]:.3f}")
        
    def modulation_error(self, spotfilter):
        self.load_mod()
        return self.spotlist.compute_modulation_error(self.mod)
    
    
    def draw_patterns(self, dims):
        for ep in tqdm.trange(len(self.mod), desc='Generating modulation pattern plots'):
            self.draw_pattern(ep,dims)
                
    def draw_pattern(self, ep, dims):
        # normalize

        ds = self.sum_ds

        mod = self.mod[ep]
        k = mod['k']
        k = k[:dims]

        sel = np.arange(len(ds))
        numpts = 2000

        np.random.seed(0)
        indices = np.arange(len(sel))
        np.random.shuffle(indices)
        sel = sel[:np.minimum(numpts,len(indices))]
        
        # Correct for phase drift
        spot_phase = self.phase_interp[ds.frame[sel]][:,ep]
        spot_phase -= np.mean(spot_phase)

        normI = self.IBg[sel][:, ep, 0] / ds.photons[sel]

        proj = (k[None] * ds.pos[sel][:,:dims]).sum(1) - spot_phase
        x = proj % (np.pi*2) 


        plt.figure(figsize=figsize)
        plt.scatter(x, normI, marker='.')

        sigx = np.linspace(0,2*np.pi,400)
        exc = mod['relint']*(1+mod['depth']*np.sin(sigx-mod['phase']))
        plt.plot(sigx, exc, 'r', linewidth=4, label='Estimated P')

        plt.ylim([-0.01,0.51])        
        plt.xlabel('Phase [radians]')
        plt.ylabel(r'Normalized intensity ($I_k$)')
        lenk = np.sqrt(np.sum(k**2))
        plt.title(f': Pattern {ep}. K={lenk:.4f} ' + f" Phase={self.mod[ep]['phase']*180/np.pi:.3f}).")
        plt.colorbar()
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.resultprefix+f"pattern{ep}.png")
        #plt.close(fig)
        

    def continuous_frc(self,maxdistance, freq=10):
                
        if self.g_undrifted is not None:
            g_data = self.g_undrifted
            sf_data = self.sf_undrifted
            print('Running continuous FRC on drift-corrected data')
        else:
            g_data = self.g2d_results
            sf_data = self.sf_results
        
        #maxdistance = 20 * np.mean(self.sum_results.get_crlb()[:,0])
        
        if np.isscalar(freq):
            freq = np.linspace(0,freq,200)
        sys.stdout.write(f'Computing continuous FRC for gaussian fits...')
        frc_g2d,val_g2d = self._cfrc(maxdistance,g_data.get_xy(),freq)
        print(f"{self.pixelsize/val_g2d:.1f} nm")
        sys.stdout.write(f'Computing continuous FRC for modulation enhanced fits...')
        frc_sf,val_sf = self._cfrc(maxdistance,sf_data.get_xy(),freq)
        print(f"{self.pixelsize/val_sf:.1f} nm")
        
        plt.figure()
        plt.plot(freq, frc_g2d, label=f'2D Gaussian (FRC={self.pixelsize/val_g2d:.1f} nm)')
        plt.plot(freq, frc_sf, label=f'Modulated (FRC={self.pixelsize/val_sf:.1f} nm)')
        plt.xlabel('Spatial Frequency (pixel^-1)')
        plt.ylabel('FRC')
        IF =  val_sf / val_g2d
        plt.title(f'Continuous FRC with localization pairs up to {maxdistance*self.pixelsize:.1f} nm distance. Improvement={IF:.2f}')
        plt.legend()
        plt.savefig(self.resultsdir+"continuous-frc.png")
        print(f"Improvement: {IF}")
        return IF
        
    def _cfrc(self, maxdist,xy,freq):
        with Context(debugMode=self.debugMode) as ctx:
            frc=PostProcessMethods(ctx).ContinuousFRC(xy, maxdist, freq, 0,0)
        
        c = np.where(frc<1/7)[0]
        v = freq[c[0]] if len(c)>0 else freq[0]
        return frc, v
        
    def _load_rois_iterator(self):
        return process_movie.load_rois_iterator(self.rois_fn)
    
    def _summed_rois_iterator(self):
        for info, pixels in process_movie.load_rois_iterator(self.rois_fn):
            yield info, pixels.sum(1)
    
    def report(self, msg):
        with open(self.reportfile,"a") as f:
            f.write(msg+"\n")
        print(msg)
        
    def simulate(self, nframes, bg, gt: Dataset, output_fn, blink_params=None, 
                 em_drift=None, exc_drift=None, debugMode=False):
        """
        Simulate SMLM dataset with excitation patterns. Very basic atm,
        spots are either off or on during the entire series of modulation patterns (6 frames for simflux)
        """
        with Context(debugMode=debugMode) as ctx, self.create_psf(ctx, True) as psf,  \
              multipart_tiff.MultipartTiffSaver(output_fn) as tif, tqdm.tqdm(total=nframes) as pb:
        
            sampleframes = psf.sampleshape[0] if len(psf.sampleshape)==3 else 1
            
            for f, spot_ontimes in enumerate(blinking_spots.blinking(len(gt), nframes // sampleframes, 
                                    **(blink_params if blink_params is not None else {}), subframe_blink=1)):
                
                
                which = np.where(spot_ontimes > 0)[0]
                
                params = np.zeros((len(which), gt.dims+2),dtype=np.float32)
                params[:,:gt.dims] = gt.pos[which]
                params[:,-2] = gt.photons[which]

                roiposYX = (params[:,[1,0]] - psf.sampleshape[-2:]/2).astype(np.int32)
                params[:,:2] -= roiposYX[:,[1,0]]
                
                roipos = np.zeros((len(which),3),dtype=np.int32)
                roipos[:,1:] = roiposYX
                
                # this way the modulation pattern will automatically be broadcasted 
                # into the required (spots, npatterns, 6) shape
                derivs, rois = psf.Derivatives(params, roipos=roipos, 
                                         constants=self.mod.view(np.float32)) 
                
                                
                for i in range(sampleframes):
                    img = np.zeros(gt.imgshape, dtype=np.float32)
                    ctx.smlm.DrawROIs(img, rois[:,i], roiposYX)
                    img += bg
                    tif.save(np.random.poisson(img))
                    pb.update()
        
    
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        



    
def view_napari(mov):
    import napari
    
    with napari.gui_qt():
        napari.view_image(mov)


def set_plot_fonts():
    import matplotlib as mpl
    new_rc_params = {
    #    "font.family": 'Times',
        "font.size": 15,
    #    "font.serif": [],
        "svg.fonttype": 'none'} #to store text as text, not as path
    mpl.rcParams.update(new_rc_params)
