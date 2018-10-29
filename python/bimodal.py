import numpy as np
import numpy.matlib as nm
import matplotlib.mlab as mlab
from svgd import SVGD

class Bimodal:
    def __init__(self, mu1, var1, mu2, var2):
        self.mu1 = mu1
        self.var1 = var1
        self.mu2 = mu2
        self.var2 = var2

    def gaussian(self, x, mu, var):
        return np.exp(-np.power(x - mu, 2.) / (2 * var))*1./(np.sqrt(2*np.pi*var))
    
    def dlnprob(self, theta):
        # return -1/2*np.log(np.exp((theta-self.mu1)*(1.0/self.var1)) + \
            # np.exp((theta-self.mu2)*(1.0/self.var2)))
        return (-1*(theta-self.mu1)*(1/self.var1)*self.gaussian(theta, self.mu1, self.var1) - \
                (theta-self.mu2)*(1/self.var2)*self.gaussian(theta, self.mu2, self.var2)) / \
                (self.gaussian(theta, self.mu1, self.var1) + self.gaussian(theta, self.mu2, self.var2))


def plot_results(mu1, var1, mu2, var2, theta, bins=20):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    count, bins, _ = plt.hist(theta, bins=bins, density=True)

    ax.set_xlim(mu1 - 3*np.sqrt(var1), mu2 + 3*np.sqrt(var2))

    x = np.linspace(mu1 - 3*np.sqrt(var1), mu2 + 3*np.sqrt(var2), 100)
    plt.plot(x, (mlab.normpdf(x, mu1, np.sqrt(var1)) + mlab.normpdf(x, mu2, np.sqrt(var2)))/2)

    plt.show()

def animate_results(mu1, var1, mu2, var2, theta_hist, n_bins=10):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.path as path
    import matplotlib.animation as animation

    fig, ax = plt.subplots()

    # histogram our data with numpy
    data = theta_hist[:, 0]
    n, bins = np.histogram(data, bins=n_bins, density=True)

    # get the corners of the rectangles for the histogram
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + n
    nrects = len(left)

    # here comes the tricky part -- we have to set up the vertex and path
    # codes arrays using moveto, lineto and closepoly

    # for each rect: 1 for the MOVETO, 3 for the LINETO, 1 for the
    # CLOSEPOLY; the vert for the closepoly is ignored but we still need
    # it to keep the codes aligned with the vertices
    nverts = nrects*(1 + 3 + 1)
    verts = np.zeros((nverts, 2))
    codes = np.ones(nverts, int) * path.Path.LINETO
    codes[0::5] = path.Path.MOVETO
    codes[4::5] = path.Path.CLOSEPOLY
    verts[0::5, 0] = left
    verts[0::5, 1] = bottom
    verts[1::5, 0] = left
    verts[1::5, 1] = top
    verts[2::5, 0] = right
    verts[2::5, 1] = top
    verts[3::5, 0] = right
    verts[3::5, 1] = bottom

    barpath = path.Path(verts, codes)
    patch = patches.PathPatch(
        barpath, facecolor='blue', edgecolor='blue', alpha=0.5)
    ax.add_patch(patch)

    ax.set_xlim(mu1 - 3*np.sqrt(var1), mu2 + 3*np.sqrt(var2))
    ax.set_ylim(bottom.min(), 0.5)

    x = np.linspace(mu1 - 3*np.sqrt(var1), mu2 + 3*np.sqrt(var2), 100)
    plt.plot(x, (mlab.normpdf(x, mu1, np.sqrt(var1)) + mlab.normpdf(x, mu2, np.sqrt(var2)))/2)

    def animate(i):
        # simulate new data coming in
        data = theta_hist[:,int(i*25)]
        n, bins = np.histogram(data, bins=n_bins, density=True)
        left = np.array(bins[:-1])
        right = np.array(bins[1:])
        bottom = np.zeros(len(left))
        top = bottom + n
        nrects = len(left)

    # here comes the tricky part -- we have to set up the vertex and path
    # codes arrays using moveto, lineto and closepoly

    # for each rect: 1 for the MOVETO, 3 for the LINETO, 1 for the
    # CLOSEPOLY; the vert for the closepoly is ignored but we still need
    # it to keep the codes aligned with the vertices
        nverts = nrects*(1 + 3 + 1)
        verts = np.zeros((nverts, 2))
        verts[0::5, 0] = left
        verts[0::5, 1] = bottom
        verts[1::5, 0] = left
        verts[1::5, 1] = top
        verts[2::5, 0] = right
        verts[2::5, 1] = top
        verts[3::5, 0] = right
        verts[3::5, 1] = bottom

        [p.remove() for p in reversed(ax.patches)]
        barpath = path.Path(verts, codes)
        patch = patches.PathPatch(
            barpath, facecolor='blue', edgecolor='blue', alpha=0.5)
        ax.add_patch(patch)

        # ax.set_xlim(left[0], right[-1])
        # ax.set_ylim(bottom.min(), top.max())

        # plt.plot(bins, 1/(np.sqrt(var) * np.sqrt(2 * np.pi)) * 
        # np.exp( - (bins - mu)**2 / (2 * np.sqrt(var)**2) ), 
        # linewidth=2, color='r')

        return [patch, ]

    ani = animation.FuncAnimation(fig, animate, int(theta_hist.shape[1]/25), repeat=False, blit=True)
    plt.show()

if __name__ == '__main__':
    mu1 = -4
    var1 = 0.5

    mu2 = 2
    var2 = 4
    
    model = Bimodal(mu1, var1, mu2, var2)
    
    x0 = np.random.normal(0,1, [200,1])
    theta, theta_hist = SVGD().update(x0, model.dlnprob, n_iter=3000, stepsize=0.01, debug=True)
    
    # print("ground truth: mu = {} var = {}".format(mu, var))
    mu = np.mean(theta)
    var = np.std(theta)**2
    print("svgd: mu = {} var = {}".format(round(mu,2), round(var,2)))

    # plot_results(mu1, var1, mu2, var2, theta)

    animate_results(mu1, var1, mu2, var2, theta_hist, n_bins=30)