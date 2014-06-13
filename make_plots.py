"""
make_plots.py
Generate plots for GALFACTS transient search. Cleans up other codes
by having these scripts separate.
04 June 2014 - Trey Wenger - Creation
06 June 2014 - Trey Wenger - Added functions for make_clusters
11 June 2014 - Joseph Kania - Editied
12 June 2014 - Trey Wenger - Allow for running on headless machine
                             by using mpl "Agg"
"""
import sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from itertools import cycle

def stokes_plot(x_data, xlabel, I_data, Q_data, U_data, V_data,
                filename):
    """Generate plot of 4 stokes parameters"""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
    ax1.plot(x_data,I_data)
    ax1.set_xlim(np.nanmin(x_data),np.nanmax(x_data))
    ax1.set_ylim(np.nanmin(I_data[I_data.nonzero()]),
                 np.nanmax(I_data))
    ax1.set_ylabel("Stokes I (K)")
    ax2.plot(x_data,Q_data)
    ax2.set_ylim(np.nanmin(Q_data),np.nanmax(Q_data))
    ax2.set_ylabel("Stokes Q (K)")
    ax3.plot(x_data,U_data)
    ax3.set_ylim(np.nanmin(U_data),np.nanmax(U_data))
    ax3.set_ylabel("Stokes U (K)")
    ax4.plot(x_data,V_data)
    ax4.set_ylim(np.nanmin(V_data),np.nanmax(V_data))
    ax4.set_ylabel("Stokes V (K)")
    ax4.set_xlabel(xlabel)
    fig.subplots_adjust(hspace=0.1)
    for ax in [ax1, ax2, ax3, ax4]:
        # make the fontsize a bit smaller
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(12)
    plt.savefig(filename)
    plt.close(fig)

def single_stokes(x_data, xlabel, y_data, ylabel, filename):
    """Generate plot of single stokes parameter"""
    fig, ax1 = plt.subplots(1)
    ax1.plot(x_data,y_data)
    ax1.set_xlim(np.nanmin(x_data),np.nanmax(x_data))
    ax1.set_ylim(np.nanmin(y_data),np.nanmax(y_data))
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    plt.savefig(filename)
    plt.close(fig)

def source_plot(dec, I_data, residuals, fit_x, fit_y, filename):
    """Generate a plot of I vs dec for a single source"""
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(dec, I_data, 'ko')
    ax1.plot(fit_x,fit_y,'k-')
    ax1.set_xlim(np.min(dec),np.max(dec))
    ax1.set_ylabel('Stokes I (K)')
    res_data = 100.*residuals/I_data
    ax2.plot(dec,res_data,'ko')
    ax2.set_ylim(-1,1)
    ax2.set_ylabel('Residuals (%)')
    ax2.set_xlabel('Dec (degs)')
    fig.subplots_adjust(hspace=0.1)
    plt.savefig(filename)
    plt.close(fig)

def field_plot(ra, dec, I_data, filename,labels=None,centers=None):
    fig, ax1 = plt.subplots(1)
    corr_ra = ra*np.cos(np.deg2rad(dec))
    if labels == None:
        sc = ax1.scatter(corr_ra,dec,c=I_data)
        cb = plt.colorbar(sc)
        cb.set_label('Stokes I (K)')
    else:
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for i,col in zip(range(len(centers)),colors):
            my_members = labels == i
            center = centers[i]
            plt.plot(corr_ra[my_members],dec[my_members],col+'o')
            plt.plot(center[0],center[1],'o',markerfacecolor=col,
                     markeredgecolor='k', markersize=14, alpha=0.5)
    plt.gca().invert_xaxis() # RA increase to left
    ax1.set_xlabel('RA * cos(Dec) (deg)')
    ax1.set_ylabel('Dec (deg)')
    plt.savefig(filename)
    plt.close(fig)

def field_plot_3d(ra, dec, I_data, fit_x, fit_y, fit_z, filename):
    fig, ax1 = plt.subplots(1,subplot_kw={"projection": '3d'})
    corr_ra = ra*np.cos(np.deg2rad(dec))
    ax1.scatter(corr_ra,dec,I_data,s=2)
    ax1.plot_wireframe(fit_x, fit_y, fit_z, rstride=10, cstride=10)
    plt.gca().invert_xaxis() # RA inreases to the left
    ax1.set_xlabel('RA * cos(Dec) (deg)')
    ax1.set_ylabel('Dec (deg)')
    ax1.set_zlabel('Stokes I (K)')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

if __name__ == "__main__":
    sys.exit("Error: module not meant to be run at top level.")
