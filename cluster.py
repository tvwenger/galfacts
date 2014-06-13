"""
source.py
Source object for GALFACTS transiet search
04 June 2014 - Trey Wenger - creation
"""
import sys
import numpy as np
import make_plots as plt
from scipy.optimize import curve_fit

class Cluster(object):
    """Cluster object for GALFACTS transient search"""
    def __init__(self, RA, DEC, AST, I_data, Q_data, U_data, V_data):
        """Initialize the source object"""
        self.RA = RA
        self.DEC = DEC
        self.AST = AST
        self.I_data = I_data
        self.Q_data = Q_data
        self.U_data = U_data
        self.V_data = V_data
        self.fit_p = None
        self.e_fit_p = None
        self.good_fit = None

    def fit(self, filename, **options):
        """Fit cluster I vs RA*cos(Dec) and Dec"""
        amp_guess = np.max(self.I_data)
        center_x_guess = self.RA[self.I_data.argmax()]
        center_y_guess = self.DEC[self.I_data.argmax()]
        # cos dec correction
        center_x_guess = center_x_guess*np.cos(np.deg2rad(center_y_guess))
        sigma_x_guess = options["beam_width"]/2.
        sigma_y_guess = options["beam_width"]/2.
        theta_guess = 0.
        guess_p = [amp_guess, center_x_guess, center_y_guess,
                   sigma_x_guess, sigma_y_guess, theta_guess]
        sigma = [options["sigma"]]*len(self.I_data)
        try:
            fit_p, covar = curve_fit(gauss2d,
                                     (self.RA*np.cos(np.deg2rad(self.DEC)),
                                      self.DEC),
                                     self.I_data, p0=guess_p,
                                     sigma=sigma)
            self.fit_p = np.array(fit_p)
            self.covar = np.array(covar)
            if (np.isinf(fit_p).any() or np.isinf(covar).any() or
                np.isnan(fit_p).any() or np.isnan(covar).any() or
                (fit_p<0).any()):
                self.good_fit = False
            else:
                self.e_fit_p = np.array([np.sqrt(covar[i,i])
                                     for i in range(len(fit_p))])
                if (np.abs(self.e_fit_p[0]/self.fit_p[0])<options["amp_req"] and
                    np.abs(self.e_fit_p[3]/self.fit_p[3])<options["width_req"] and
                    np.abs(self.e_fit_p[4]/self.fit_p[4])<options["width_req"]):
                    self.good_fit = True
                else:
                    self.good_fit = False
                # for plotting
                if options["file_verbose"]:
                    fit_x = np.linspace(np.min(self.RA),
                                        np.max(self.RA), 100)
                    fit_y = np.linspace(np.min(self.DEC),
                                        np.max(self.DEC), 100)
                    fit_x = fit_x * np.cos(np.deg2rad(fit_y))
                    mesh_x, mesh_y = np.meshgrid(fit_x, fit_y)
                    fit_z = np.array([[gauss2d((x,y),*self.fit_p)
                                       for x in fit_x]
                                       for y in fit_y])
                    plt.field_plot_3d(self.RA, self.DEC, self.I_data,
                                      mesh_x, mesh_y, fit_z, filename)
        except RuntimeError:
            if options["verbose"]:
                print("Log: A fit did not converge.")
            self.good_fit = False
            

def gauss2d((x, y), *p):
    amp, center_x, center_y, sigma_x, sigma_y, theta = p
    a = np.cos(theta)**2./(2.*sigma_x**2.)
    a = a + np.sin(theta)**2./(2.*sigma_y**2.)
    b = -np.sin(2.*theta)/(4.*sigma_x**2.)
    b = b + np.sin(2.*theta)/(4*sigma_y**2.)
    c = np.sin(theta)**2./(2.*sigma_x**2.)
    c = c + np.cos(theta)**2./(2.*sigma_y**2.)
    arg = a*(x - center_x)**2.
    arg = arg + 2.*b*(x - center_x)*(y - center_y)
    arg = arg + c*(y - center_y)**2.
    return amp * np.exp(-arg)
    
if __name__ == "__main__":
    sys.exit("Error: module not mean to be run from top level")
