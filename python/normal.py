import numpy as np
import numpy.matlib as nm
from svgd import SVGD

class MVN:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    
    def dlnprob(self, theta):
        return -1*(theta-nm.repmat(self.mu, theta.shape[0], 1) * self.sigma)

def plot_results(mu, sigma, theta):
    import matplotlib.pyplot as plt
    
    count, bins, _ = plt.hist(theta, 50, density=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * 
        np.exp( - (bins - mu)**2 / (2 * sigma**2) ), 
        linewidth=2, color='r')

    plt.show()

if __name__ == '__main__':
    sigma = np.array([0.2260])
    mu = np.array([-0.6871])
    
    model = MVN(mu, sigma)
    
    x0 = np.random.normal(0,1, [100,1]);
    theta = SVGD().update(x0, model.dlnprob, n_iter=1000, stepsize=0.01)
    
    print("ground truth: ", mu)
    print("svgd: ", np.mean(theta,axis=0))

    plot_results(mu, sigma, theta)