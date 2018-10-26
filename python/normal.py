import numpy as np
import numpy.matlib as nm
from svgd import SVGD

class MVN:
    def __init__(self, mu, var):
        self.mu = mu
        self.var = var
    
    def dlnprob(self, theta):
        return -1*(theta-self.mu)*(1.0/self.var)

def plot_results(mu, var, theta, bins=20):
    import matplotlib.pyplot as plt

    count, bins, _ = plt.hist(theta, bins=bins, density=True)
    plt.plot(bins, 1/(np.sqrt(var) * np.sqrt(2 * np.pi)) * 
        np.exp( - (bins - mu)**2 / (2 * np.sqrt(var)**2) ), 
        linewidth=2, color='r')

    plt.show()

def animate_results(my, var, theta_hist, n_bins=20):
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

    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(bottom.min(), top.max())

    plt.plot(bins, 1/(np.sqrt(var) * np.sqrt(2 * np.pi)) * 
        np.exp( - (bins - mu)**2 / (2 * np.sqrt(var)**2) ), 
        linewidth=2, color='r')

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

        ax.set_xlim(left[0], right[-1])
        ax.set_ylim(bottom.min(), top.max())

        plt.plot(bins, 1/(np.sqrt(var) * np.sqrt(2 * np.pi)) * 
        np.exp( - (bins - mu)**2 / (2 * np.sqrt(var)**2) ), 
        linewidth=2, color='r')

        return [patch, ]

    ani = animation.FuncAnimation(fig, animate, int(theta_hist.shape[1]/25), repeat=False, blit=False)
    plt.show()

if __name__ == '__main__':
    var = 1
    mu = 5
    
    model = MVN(mu, var)
    
    x0 = np.random.normal(0,1, [150,1])
    theta, theta_hist = SVGD().update(x0, model.dlnprob, n_iter=1000, stepsize=0.01, debug=True)
    
    print("ground truth: mu = {} var = {}".format(mu, var))
    print("svgd: mu = {} var = {}".format(round(np.mean(theta),2), round(np.std(theta)**2,2)))

    # plot_results(mu, var, theta)

    animate_results(mu, var, theta_hist)