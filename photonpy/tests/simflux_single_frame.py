"""
Testing Single-frame simflux models
"""
import numpy as np
import matplotlib.pyplot as plt

import photonpy.smlm.util as su
from photonpy.cpp.context import Context
import photonpy.cpp.gaussian as gaussian

import photonpy.cpp.com as com 
from photonpy.cpp.simflux import SIMFLUX
import tqdm

def plot_traces(traces, theta, psf, axis):
    plt.figure()
    theta=np.array(theta)
    for k in range(len(traces)):
        plt.plot(np.arange(len(traces[k])),traces[k][:,axis])
        plt.plot([len(traces[k])],[theta[k,axis]],'o')
    plt.title(f"Axis {axis}[{psf.param_names[axis]}]")


def crlb_plot(data, photons):
    axes=['x', 'I', 'bg']
    axes_unit=['pixels', 'photons','photons/pixel']
    axes_scale=[1, 1, 1, 1]
    for i,ax in enumerate(axes):
        plt.figure()
        for psf,name,(prec,bias,crlb) in data:
            ai = psf.ParamIndex(ax)
            line, = plt.gca().plot(photons,axes_scale[i]*prec[:,ai],label=f'Precision {name}')
            plt.plot(photons,axes_scale[i]*crlb[:,ai],label=f'CRLB {name}', color=line.get_color(), linestyle='--')

        plt.title(f'{ax} axis')
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel('Signal intensity [photons]')
        plt.ylabel(f"{ax} [{axes_unit[i]}]")
        plt.grid(True)
        plt.legend()

        plt.figure()
        for psf,name,(prec,bias,crlb) in data:
            ai = psf.ParamIndex(ax)
            plt.plot(photons,bias[:,ai],label=f'Bias {name}')

        plt.title(f'{ax} axis')
        plt.grid(True)
        plt.xscale("log")
        plt.legend()
        plt.show()


def estimate_precision(sample_psf, estimate_fn,crlb_fn, thetas, photons):
    prec = np.zeros((len(photons),thetas.shape[1]))
    bias = np.zeros((len(photons),thetas.shape[1]))
    crlb = np.zeros((len(photons),thetas.shape[1]))
    
    Iidx = psf.ParamIndex('I')
     
    for i in tqdm.trange(len(photons)):
        thetas_ = thetas*1
        thetas_[:, Iidx] = photons[i]
        
        roipos = np.random.randint(0,20,size=(len(thetas_), sample_psf.indexdims))
        roipos[:,0] = 0
        smp = sample_psf.GenerateSample(thetas_,roipos=roipos)

        estim,diag,traces = estimate_fn(smp,roipos=roipos)
            
        if i == len(photons)-1:
            plot_traces(traces[:20], thetas_[:20], psf, axis=0)
            plot_traces(traces[:20], thetas_[:20], psf, axis=2)
            plot_traces(traces[:20], thetas_[:20], psf, axis=Iidx)

        crlb_ = crlb_fn(thetas_,roipos=roipos)
        err = estim-thetas_
        prec[i] = np.std(err,0)
        bias[i] = np.mean(err,0)
        crlb[i] = np.mean(crlb_,0)

#    print(f'sigma bias: {bias[:,4]}')        
    return prec,bias,crlb

def test_template_fit(offsets, psf, mod):
    """ 
    Test the estimation of all the spot intensities separately (simfluxMode=False)
    """ 
    sf_estim = SIMFLUX(psf.ctx).CreateSFSFEstimator(psf, mod, offsets, simfluxMode=False)
    
    print(sf_estim.sampleshape)
    print(sf_estim.ParamFormat())

    roisize = psf.sampleshape[-1]
    sf_theta=[[roisize//2, roisize//2, *np.linspace(0,400,len(mod)), 5]]

    ev = sf_estim.ExpectedValue(sf_theta)  

    smp = np.random.poisson(ev) 

    # Center of ROI + spread all photons evenly over the spots
    initial_theta=[[roisize//2, roisize//2-4, *(np.ones(len(mod)) * np.sum(smp)/len(mod)), 0]]
    estim,_,traces = sf_estim.Estimate(smp, initial=initial_theta)
    print(estim[0])

    plot_traces(traces, sf_theta, psf, 0)
    plot_traces(traces, sf_theta, psf, 2)

    estim_ev = sf_estim.ExpectedValue(estim)    
    plt.figure()
    plt.imshow(np.concatenate((smp[0], estim_ev[0]),-1))
    plt.colorbar()
    plt.title("Sample (left) vs expected value for estimate (right)")
    
def test_simflux_fit(offsets, psf, mod):
    sf = SIMFLUX(psf.ctx)
    sf_estim = sf.CreateSFSFEstimator(psf, mod, offsets, simfluxMode=True)
    sf_estim.SetLevMarParams(1e-15,50)

    print(f"Testing simflux estimator:")    
    print(sf_estim.sampleshape)
    print(sf_estim.ParamFormat())
    
    print(sf_estim.limits)

    roisize = psf.sampleshape[-1]
    sf_theta=[[roisize//2-1, roisize//2-2, 1000, 5]]

    deriv,ev = sf_estim.Derivatives(sf_theta)
    deriv=deriv[0]
          
    for k in range(len(deriv)):
        plt.figure()
        plt.imshow(deriv[k])
        plt.title(sf_estim.param_names[k])
        plt.colorbar()

    smp = np.random.poisson(ev) 
    
    # Center of ROI + spread all photons evenly over the spots
    print(f'Sum(smp):{np.sum(smp)}')
    initial_theta=[[roisize//2, roisize//2, np.sum(smp), 0]]
    print(initial_theta)
    estim,_,traces = sf_estim.Estimate(smp, initial=initial_theta)
        
    print(estim[0])

    plot_traces(traces, sf_theta, psf, 0)
    plot_traces(traces, sf_theta, psf, 2)

    estim_ev = sf_estim.ExpectedValue(estim)    
    plt.figure()
    plt.imshow(np.concatenate((smp[0], estim_ev[0]),-1))
    plt.colorbar()
    plt.title("Sample (left) vs expected value for estimate (right)")
    
def plot_crlb_vs_r(psf, mod):
    r_range = np.linspace(0,10,30)
    
    numspots = 500
    theta = np.zeros((numspots, 4))
    theta[:,[0,1]] = np.random.uniform(-4,4,size=(numspots,2)) + roisize/2
    theta[:,2] = 1000
    theta[:,3] = 1

    bglist = [1, 8]

    crlb = np.zeros((len(r_range), len(bglist), 2, 4))
    
    for i in tqdm.trange(len(r_range)):
        R=r_range[i]
        for bgi, bg in enumerate(bglist):
            offsets = np.zeros((len(mod),2))
            ang = np.linspace(0,2*np.pi, len(mod), endpoint=False)
            offsets[:,0] = R*np.cos(ang)
            offsets[:,1] = R*np.sin(ang)

            with Context(smlm=psf.ctx.smlm) as tempctx: 
                sf = SIMFLUX(tempctx)
                sfsf_estim = sf.CreateSFSFEstimator(psf, mod, offsets, simfluxMode=True)

                theta_bg = theta*1
                theta_bg[:,3] = bg            

                #ie = sf.CreateSFSFEstimator(psf, mod*0, offsets,simfluxMode=False)
                #ci = ie.SumIntensities(ie.CRLB(ie.ExpandIntensities(theta_bg)))
                #crlb[i,bgi,1]=np.mean(ci,0)

                c = sfsf_estim.CRLB(theta_bg)
                crlb[i,bgi,0]=np.mean(c,0)
    
    plt.figure()
    lines=[]
    for bgi in range(len(bglist)):
        lines.append(plt.plot(r_range, crlb[:,bgi,0,0], label=f"SNIPE, bg={bglist[bgi]} photons/px")[0])
        #plt.plot(r_range, crlb[:,bgi,1,0], label=f"Independent intensities, bg={bglist[bgi]} photons/px")

    for bgi, bg in enumerate(bglist):
        theta_bg = np.array([theta[0]])
        theta_bg[:,3] = bg
        crlbx = psf.CRLB(theta_bg)[:,0]
        plt.gca().axhline(crlbx, color=lines[bgi].get_color(),
                          linestyle='--', label=f"2D Gaussian, bg={bglist[bgi]} photons/px")

    plt.legend()    
    plt.xlabel('Radius of SNIPE offsets [pixels]')
    plt.ylabel('Precision in X [pixels]')
    plt.title('Offset radius vs CRLB X')
 
    
def test_crlb(offsets, psf, mod):
    roisize = psf.sampleshape[-1]
    numspots = 500

    theta = np.zeros((numspots, 4))
    theta[:,[0,1]] = np.random.uniform(-4,4,size=(numspots,2)) + roisize/2
    theta[:,2] = 1000
    theta[:,3] = 1
    
    sf_theta = theta*1 # for SIMFLUX, we create a set of parameter with 6x the background, to make it fair
    sf_theta[:,3] *= len(mod)
    
    photons = np.logspace(2, 5, 20)
    
    com_estim = com.CreateEstimator(roisize, ctx)

    sf = SIMFLUX(psf.ctx)
    initial_estim = sf.CreateSFSFEstimator(psf, mod, offsets,simfluxMode=False)
    sfsf_estim = sf.CreateSFSFEstimator(psf, mod, offsets, simfluxMode=True)
    sf_estim = sf.CreateEstimator_Gauss2D(psf.calib, mod, roisize, len(offsets), simfluxEstim=True)
    
    plt.figure()
    plt.imshow(np.concatenate(sf_estim.GenerateSample(sf_theta[0])[0],-1))
    plt.title('SIMFLUX Sample')        
    
    plt.figure()
    plt.imshow(sfsf_estim.GenerateSample(theta[0])[0])
    plt.title('SNIPE Sample')
    
    def sf_estimate(samples, roipos):
        return sf_estim.Estimate(samples,roipos=roipos)
    
    def sfsf_init_estimate(samples, roipos):
        # COM Estimate -> SFSF non-simflux fit -> SFSF simflux fit -> result
        initial = com_estim.Estimate(samples)[0]
        initial = initial_estim.ExpandIntensities(initial)
        estim,diag,traces = initial_estim.Estimate(samples, roipos=roipos, initial=initial)
        
        return initial_estim.SumIntensities(estim), diag, traces

    def sfsf_init_crlb(theta,roipos):
        theta2 = initial_estim.ExpandIntensities(theta)
        crlb = initial_estim.CRLB(theta2)
        # ignoring intensity cross terms but those are probably very small
        return np.sqrt(1/initial_estim.SumIntensities(1/crlb**2))

    def sfsf_estimate(samples, roipos):
        sfsf_initial = sfsf_init_estimate(samples,roipos)[0]
        return sfsf_estim.Estimate(samples, roipos, initial=sfsf_initial)

    if True:
        data = [
            (sf_estim, 'Regular SIMFLUX', 
                 estimate_precision(sf_estim, sf_estim.Estimate, sf_estim.CRLB, sf_theta, photons)),
            (sfsf_estim, 'SNIPE Initializer', 
                 estimate_precision(sfsf_estim, sfsf_init_estimate, sfsf_init_crlb, theta, photons)),
            (sfsf_estim, 'SNIPE', 
                 estimate_precision(sfsf_estim, sfsf_estimate, sfsf_estim.CRLB, theta, photons))
            ]
    
        crlb_plot(data, photons)

images=[]

def debugImage(img,label):
    images.append((img,label))


def plotDebugImages():
    for img,label in images:
        plt.figure()
        plt.imshow( np.concatenate(img,-1) )
        plt.colorbar()
        plt.title(label)


with Context(debugMode=False) as ctx:
    g = gaussian.Gaussian(ctx)

    roisize=28
    sigma = 1.4
    
    psf = g.CreatePSF_XYIBg(roisize, sigma, cuda=True)

    ctx.smlm.SetDebugImageCallback(debugImage)

    mod = np.array([
         [0, 1.8, 0, 0.95, 0, 1/6],
         [1.8, 0, 0, 0.95, 0, 1/6],
         [0, 1.8, 0, 0.95, 2*np.pi/3, 1/6],
         [1.8, 0, 0, 0.95, 2*np.pi/3, 1/6],
         [0, 1.8, 0, 0.95, 4*np.pi/3, 1/6],
         [1.8, 0, 0, 0.95, 4*np.pi/3, 1/6]
    ])


    offsets = np.zeros((len(mod),2))
    ang = np.linspace(0,2*np.pi, len(mod), endpoint=False)
    R = 7
    offsets[:,0] = R*np.cos(ang)
    offsets[:,1] = R*np.sin(ang)


    #test_template_fit(offsets, psf, mod)
        
    #test_simflux_fit(offsets, psf, mod)
    
    #plotDebugImages()
    
    #test_crlb(offsets, psf, mod)
    
    plot_crlb_vs_r(psf,mod)


    """
    deriv, ev = sf_estim.Derivatives(sf_theta)
    fig,ax=plt.subplots(1,deriv.shape[1],sharey=True)
    for k in range(deriv.shape[1]):
        ax[k].imshow(deriv[0,k])
        ax[k].set_title(sf_estim.param_names[k])
                
    """
    
    