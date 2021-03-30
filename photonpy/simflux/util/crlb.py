import numpy as np
import scipy.special as sps



# returns Ex, Ey, D(Ex,x), D(Ey,y)
def gauss2D_derivatives(theta, numpixels, psfSigma, testplot=False):
    x,y = theta[0],theta[1]
    pixelpos = np.arange(0, numpixels)

    OneOverSqrt2PiSigma = 1.0 / (np.sqrt(2 * np.pi) * psfSigma)
    OneOverSqrt2Sigma = 1.0 / (np.sqrt(2) * psfSigma)

    # Pixel centers
    Xc, Yc = np.meshgrid(pixelpos, pixelpos)
    Xexp0 = (Xc-x+0.5) * OneOverSqrt2Sigma 
    Xexp1 = (Xc-x-0.5) * OneOverSqrt2Sigma 
    Ex = 0.5 * sps.erf(Xexp0) - 0.5 * sps.erf(Xexp1)
    dEx = OneOverSqrt2PiSigma * (np.exp(-Xexp1**2) - np.exp(-Xexp0**2))

    Yexp0 = (Yc-y+0.5) * OneOverSqrt2Sigma  
    Yexp1 = (Yc-y-0.5) * OneOverSqrt2Sigma
    Ey = 0.5 * sps.erf(Yexp0) - 0.5 * sps.erf(Yexp1)
    dEy = OneOverSqrt2PiSigma * (np.exp(-Yexp1**2) - np.exp(-Yexp0**2))
    
    return Ex, Ey, dEx, dEy


def jacobian_silm_XYIBg(theta, roisize, sigma, mod):
    gEx,gEy,gdEx,gdEy=gauss2D_derivatives(theta, roisize, sigma)
    
    gEx = np.tile(gEx,(len(mod),1,1))
    gEy = np.tile(gEy,(len(mod),1,1))
    gdEx = np.tile(gdEx,(len(mod),1,1))
    gdEy = np.tile(gdEy,(len(mod),1,1))
    
    Q = (1+mod[:,2] * np.sin(mod[:,0]*theta[0]+mod[:,1]*theta[1]-mod[:,3]))*mod[:,4]
    dQdx = mod[:,4] * mod[:,2] * mod[:,0] * np.cos(mod[:,0]*theta[0]+mod[:,1]*theta[1]-mod[:,3])
    dQdy = mod[:,4] * mod[:,2] * mod[:,1] * np.cos(mod[:,0]*theta[0]+mod[:,1]*theta[1]-mod[:,3])
    
    Q = Q[:,np.newaxis,np.newaxis]
    dQdx = dQdx[:,np.newaxis,np.newaxis]
    dQdy = dQdy[:,np.newaxis,np.newaxis]
    
    gExy = gEx*gEy
    
    mu = theta[2] * (Q*gExy) + theta[3] / len(mod)
    dmu_x = theta[2] * (dQdx * gExy + Q * gdEx * gEy)
    dmu_y = theta[2] * (dQdy * gExy + Q * gEx * gdEy)
    dmu_I = Q*gExy
    dmu_bg = 1/len(mod)
    
    return mu, [dmu_x,dmu_y,dmu_I,dmu_bg]

def jacobian_silm_XYI(theta,roisize,sigma,mod):
    mu,jac = jacobian_silm_XYIBg(theta,roisize,sigma,mod)
    return mu,jac[:-1]

def jacobian_XYIBg(theta, roisize, sigma):
    Ex,Ey,dEx,dEy=gauss2D_derivatives(theta, roisize, sigma)

    mu = theta[2] * Ex*Ey + theta[3]
    dmu_x = theta[2] * Ey * dEx
    dmu_y = theta[2] * Ex * dEy    
    dmu_I = Ex*Ey
    dmu_bg = 1
    return mu, [dmu_x, dmu_y, dmu_I, dmu_bg]

def jacobian_XYI(theta, roisize, sigma):
    Ex,Ey,dEx,dEy=gauss2D_derivatives(theta, roisize, sigma)

    mu = theta[2] * Ex*Ey + theta[3]
    dmu_x = theta[2] * Ey * dEx
    dmu_y = theta[2] * Ex * dEy    
    dmu_I = Ex*Ey
    return mu, [dmu_x, dmu_y, dmu_I]

def compute_crlb(mu, jac):
    mu[mu<1e-9] = 1e-9
    K = len(jac)
    fi = np.zeros((K,K))
    for i in range(K):
        for j in range(K):            
            fi[i,j] = np.sum( 1/mu * (jac[i] * jac[j]))
    return np.sqrt(np.linalg.inv(fi).diagonal())
