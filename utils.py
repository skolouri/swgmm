import numpy as np
import scipy
from scipy import interp
from scipy.ndimage import filters
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

def pWasserstein(I0,I1,p):
    """Given two one-dimensional pdfs I_0 and I_1, this function calculates the following:
    
    f:   Transport map between I0 and I1, such that f'I_1(f)=I_0
    phi: The transport displacement potential f(x)=x-\nabla phi(x)
    Wp:  The p-Wasserstein distance
    """
    assert I0.shape==I1.shape
    eps=1e-7
    I0=I0+eps # Add a small value to pdfs to ensure positivity everywhere
    I1=I1+eps
    I0=I0/I0.sum() # Normalize the inputs to ensure that they are pdfs
    I1=I1/I1.sum()
    J0=np.cumsum(I0) # Calculate the CDFs
    J1=np.cumsum(I1)
    # Here we calculate transport map f(x)=x-u(x) 
    x=np.asarray(range(len(I0))) 
    xtilde=np.linspace(0,1,len(I0))
    XI0 = interp(xtilde,J0, x)
    XI1 = interp(xtilde,J1, x)
    u = interp(x,XI0,XI0-XI1) # u(x)
    f = x-u
    phi= np.cumsum(u/(len(I0))) # Integrate u(x) to obtain phi(x)
    phi-=phi.mean() # Subtract the mean of phi to account for the unknown constant
    Wp=(((abs(u)**p)*I0).mean())**(1.0/p)
    return f,phi, Wp 

def gaussKernel(t,proj):    
    density=np.histogram(proj,bins=len(t),range=(t.min(),t.max()))[0]
#     filters.gaussian_filter1d(density,sigma=rho)
    return density/float(density.sum())

def gaussKernelEM(t,proj,rho=.001):    
    density=((1.0/np.sqrt(2*np.pi*(rho**2)))*np.exp(-((np.tile(proj,(t.shape[0],1))-np.tile(t,(proj.shape[0],1)).T)**2)/(2*rho**2))).sum(1)
    return density/density.sum()

def gaussian1d(t,mu,sigma):
    return (1.0/(sigma*np.sqrt(2*np.pi)))*np.exp(-((t-mu)**2)/(2*sigma**2))

def gmm1d(t,mu,sigma,alpha):
    out=np.zeros_like(t)
    for i in range(len(mu)):
        out+=alpha[i]*gaussian1d(t,mu[i],sigma[i])
    return out/out.sum()


def gauss2D(X,mu,Sigma):
    P = np.linalg.det(Sigma) ** -.5 * (2 * np.pi) ** (-X.shape[1]/2.) \
                * np.exp(-.5 * np.einsum('ij, ij -> i',\
                        X - mu, np.dot(np.linalg.inv(Sigma) , (X - mu).T).T ) ) 
    return P

def sample_gmm(alphas,means,sigmas,N):    
    out=np.zeros((N,))
    label=np.zeros((N,))
    for n in range(N):
        ind=np.random.choice(np.arange(len(alphas)),p=alphas)
        out[n]=np.random.normal(loc=means[ind],scale=sigmas[ind])
        label[n]=ind
    return out,label

def generateTheta(L,d):
    theta=np.zeros((L,d))
    th_=np.random.rand(1,d)
    theta[0,:]=th_/np.sqrt((th_**2).sum())
    for i in range(1,L):
        th_=np.random.randn(1,d)
        th_=th_/np.sqrt((th_**2).sum())
        m=abs(np.matmul(theta[:i,:],th_.T)).max()
        while m>0.97:
            th_=np.random.randn(1,d)
            th_=th_/np.sqrt((th_**2).sum())
            m=abs(np.matmul(theta[:i,:],th_.T)).max()
        theta[i,:]=th_ 
    return theta

def plot_cov_ellipse(pos,cov, nstd=2, ax=None):    
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, facecolor='none',edgecolor='k',linewidth=2,zorder=25)

    ax.add_artist(ellip)
    return ellip

def plot_angles(theta):
    ax = plt.gca()
    x=np.concatenate((np.linspace(-1,1,1000),np.linspace(-1,1,1000)))
    y=np.concatenate((np.sqrt(1-np.linspace(-1,1,1000)),-np.sqrt(1-np.linspace(-1,1,1000))))
    ax.plot(x,y,linewidth=3)
    ax.plot(theta[:,0],theta[:,1],'x',linewidth=3)
def logLikelihood(X,mu_,Sigma_,alpha_):
    epsilon=1e-10
    K=len(alpha_)
    N=X.shape[0]
    R = np.zeros((N, K))
    for k in range(K):
        R[:, k] = alpha_[k] * gauss2D(X,mu_[k],Sigma_[k])
    return np.mean(-np.log(R.sum(axis=1)))

def swdistance(X,mu_,Sigma_,alpha_,L=180):
    N,d=X.shape
    K=len(mu_)
    t=np.linspace(-np.abs(X).max()*np.sqrt(2*d),np.abs(X).max()*np.sqrt(2*d),1000)
    theta=np.zeros((L,d))
    rho=np.linspace(0,180,L)*np.pi/180.0
    theta[:,0]=np.cos(rho)
    theta[:,1]=np.sin(rho)
    xproj=np.matmul(X,theta.T)
    projectedSigma=np.zeros((K,L))
    projectedMu=np.zeros((K,L))
    for k,(sig,m) in enumerate(zip(Sigma_,mu_)):
        for l,th in enumerate(theta):        
            projectedSigma[k,l]=np.sqrt(np.matmul(np.matmul(th,sig),th))
            projectedMu[k,l]=np.matmul(th,m)        
    sw=0
    for l in range(L):
        RIx=gmm1d(t,projectedMu[:,l],projectedSigma[:,l],alpha_)
        RIy=gaussKernel(t,xproj[:,l])
        _,_,w2=pWasserstein(RIx,RIy,p=2)
        sw+= w2/float(L)
    return sw

def projectPD(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q@xdiag@Q.T