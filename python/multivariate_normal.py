import numpy as np
import numpy.matlib as nm
from svgd import SVGD

class MVN:
    def __init__(self, mu, A):
        self.mu = mu
        self.A = A
    
    def dlnprob(self, theta):
        return -1*np.matmul(theta-nm.repmat(self.mu, theta.shape[0], 1), self.A)


def plot_results(mu, A, theta, bins=20):
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal
    from mpl_toolkits.mplot3d import Axes3D
    
    #Create grid and multivariate normal
    x = np.linspace(-2,6,500)
    y = np.linspace(-4,4,500)
    X, Y = np.meshgrid(x,y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    rv = multivariate_normal(mu, A)

    #Make a 3D plot
    fig_1 = plt.figure()
    ax = fig_1.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    [xmin01, xmax01, ymin01, ymax01] = ax.axis() 

    # Plot theta
    fig_2 = plt.figure()
    ax2 = fig_2.gca(projection='3d')
    ax2.set_xlim(left=xmin01, right=xmax01)
    ax2.set_ylim(bottom=ymin01, top=ymax01)

    hist, xedges, yedges = np.histogram2d(theta[:,0], theta[:,1], bins=bins, normed=True)
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()

    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    ax2.set_zlabel('Z axis')

    plt.show()

if __name__ == '__main__':
    A = np.array([[0.2260,0.1652],[0.1652,0.6779]])
    mu = np.array([-0.6871,2])
    
    model = MVN(mu, A)
    
    x0 = np.random.normal(0,1, [100,2]);
    theta = SVGD().update(x0, model.dlnprob, n_iter=100000, stepsize=0.01, debug=True)
  
    print("ground truth: ", mu)
    print("svgd: ", np.mean(theta,axis=0))

    plot_results(mu, A, theta, bins=10)