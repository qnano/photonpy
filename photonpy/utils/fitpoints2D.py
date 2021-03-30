import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as gaussian_kde
import scipy.spatial.ckdtree

def fitlsq(x,y,z,imgsize,gridshape):
    """
    Fits a 2D quadratic function z = f(x,y) to the given points,
    returning the image and the coefficients
    """
    def compute_A(X,Y):
        return np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, 
                      Y**2, X*Y**2, X*Y]).T

    A = compute_A(x,y)
    coeff, r, rank, s = np.linalg.lstsq(A, z,rcond=None)
    
    X = np.linspace(0, imgsize[1], gridshape[1])
    Y = np.linspace(0, imgsize[0], gridshape[0])
    X,Y = np.meshgrid(X,Y)
    
    A = compute_A(X.flatten(),Y.flatten())
    fit = (A @ coeff).reshape(gridshape)
    
    return fit, coeff

def hist2d_statistic_points(x,y,z, imgsize, gridshape, statistic='mean'):
    
    means,_,_,binnr  = scipy.stats.binned_statistic_2d(x,y,z, bins=gridshape, statistic=statistic, 
                                           range=[[0,imgsize[1]],[0,imgsize[0]]])

    
    indices = np.array(np.nonzero((~np.isnan(means)))).T
    xy = ( (indices+0.5)/gridshape)*imgsize
    
    return np.array ([ xy[:,0],xy[:,1], means[~np.isnan(means)] ]).T
        
def find_groups_xy(xy, searchdist):
    kdtree = scipy.spatial.ckdtree.cKDTree(xy)
    pairs = kdtree.query_pairs(searchdist) # output_type='ndarray')
    print(f"query_pairs returned {len(pairs)} pairs")

    group_indices = [-1]*len(xy)
    group_sum_x = [0]*len(xy)
    group_sum_y = [0]*len(xy)
    group_counts = [0]*len(xy)
    group_index = 0
    
    for a,b in pairs:
        if group_indices[a] < 0:
            # Allocate new group
            group_indices[a] = group_index
            group_indices[b] = group_index
            i = group_index
            group_sum_x[i] += xy[a,0]
            group_sum_y[i] += xy[a,1]
            group_counts[i] += 1
            group_index += 1
        else:
            group_indices[b] = group_indices[a]
            i = group_indices[a]

        group_counts[i] += 1
        group_sum_x[i] += xy[b,0]
        group_sum_y[i] += xy[b,1]
        
    for i in range(len(xy)):
        if group_indices[i]<0:
            # still no group
            group_sum_x[group_index] += xy[a,0]
            group_sum_y[group_index] += xy[a,1]
            group_indices[i] = group_index
            group_counts[group_index] += 1
            group_index += 1

    counts = np.array(group_counts[:group_index])
    means = np.array([group_sum_x[:group_index],group_sum_y[:group_index]]).T / counts[:,np.newaxis]
    return means, group_indices, np.array(group_counts[:group_index])

    
def gauss_smooth(xp,yp,zp,imgsize,gridshape,sigma=20):
#    kernel = gaussian_kde( np.array([xp,yp]), sigma )
    def compute(xpos,ypos):
        prob = np.exp(- ((xp-xpos) ** 2 + (yp-ypos) ** 2) / (2*sigma**2) ) / (sigma**2*2*np.pi)
        return np.sum( zp * prob, -1) / np.sum(prob,-1)

    X = np.linspace(0, imgsize[1], gridshape[1])
    Y = np.linspace(0, imgsize[0], gridshape[0])

    vals = np.zeros(gridshape)
    for b,y in enumerate(Y):
        for a,x in enumerate(X):
            vals[b,a] = compute(x,y)
    
    return vals
    
if __name__ == "__main__":
    pts = np.random.uniform(np.array([0,0,-5]),np.array([200,200,5]),size=(50,3))
    pts[:,2] = 2e-4 * pts[:,0] ** 2 - 1e-4 * pts[:,1] ** 2
    
    imgsize,gridshape = [200,200],[200,200]
    img, coeff = fitlsq(pts[:,0],pts[:,1],pts[:,2],imgsize,gridshape)
 
    img_gauss = gauss_smooth(pts[:,0],pts[:,1],pts[:,2],imgsize,gridshape,sigma=20)
    plt.figure()
    plt.imshow(img_gauss)
    plt.scatter(pts[:,0],pts[:,1], c=pts[:,2])
    plt.colorbar()
    
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
#    plt.scatter(pts[:,0],pts[:,1], c=pts[:,2])
    
    
    xyz = hist2d_statistic_points(pts[:,0],pts[:,1],pts[:,2], imgsize, [10,10])
    plt.scatter(xyz[:,0],xyz[:,1],c=xyz[:,2])

    print(np.mean( xyz[:,2]))
    