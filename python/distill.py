import numpy as np
import numpy.matlib as nm
import matplotlib.mlab as mlab
from svgd import SVGD, SVGD_Distill

class MVN:
    def __init__(self, mu, var):
        self.mu = mu
        self.var = var
    
    def dlnprob(self, theta):
        return -1*(theta-self.mu)*(1.0/self.var)

def plot_results(mu1, var1, mu2, var2, theta, bins=20):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    count, bins, _ = plt.hist(theta, bins=bins, density=True)

    x = np.linspace(mu1 - 3*np.sqrt(var1), mu2 + 3*np.sqrt(var2), 100)
    plt.plot(x, mlab.normpdf(x, mu1, np.sqrt(var1)))
    plt.plot(x, mlab.normpdf(x, mu2, np.sqrt(var2)))

    plt.show()

def animate_results(mu1, var1, mu2, var2, theta_hist, n_bins=20):
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
    ax.set_ylim(0, 0.5)

    x = np.linspace(mu1 - 3*np.sqrt(var1), mu2 + 3*np.sqrt(var2), 100)
    plt.plot(x, mlab.normpdf(x, mu1, np.sqrt(var1)))
    plt.plot(x, mlab.normpdf(x, mu2, np.sqrt(var2)))

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

        return [patch, ]

    ani = animation.FuncAnimation(fig, animate, int(theta_hist.shape[1]/25), repeat=False, blit=False)
    plt.show()

if __name__ == '__main__':
    mu1 = -4
    var1 = 1

    mu2 = 2
    var2 = 5

    model1 = MVN(mu1, var1)
    model2 = MVN(mu2, var2)
    
    x0 = np.random.normal(0,1, [150,1])
    theta, theta_hist = SVGD_Distill().update(x0, model1.dlnprob, model2.dlnprob, n_iter=1000, stepsize=0.01, debug=True)
    
    # print("ground truth: mu = {} var = {}".format(mu, var))
    # print("svgd: mu = {} var = {}".format(round(np.mean(theta),2), round(np.std(theta)**2,2)))

    # plot_results(mu1, var1, mu2, var2, theta)

    animate_results(mu1, var1, mu2, var2, theta_hist)