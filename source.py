"""
source.py
Source object for GALFACTS transiet search
04 June 2014 - Trey Wenger - creation
11 June 2014 - Joseph Kania - Modification
"""
import sys
import numpy as np
import make_plots
from scipy.optimize import curve_fit

class Source(object):
    """Source object for GALFACTS transient search"""
    def __init__(self, RA, DEC, AST, I_data, Q_data, U_data, V_data,
                 time_end, dec_end):
        """Initialize the source object"""
        self.RA = RA
        self.DEC = DEC
        self.AST = AST
        self.I_data = I_data
        self.Q_data = Q_data
        self.U_data = U_data
        self.V_data = V_data
        self.time_end = time_end
        self.dec_end = dec_end
        self.fit_p = None
        self.e_fit_p = None
        self.good_fit = None
        self.center_RA = None
        self.center_DEC = None
        self.I_baselined = None
        self.center_I = None

    def fit(self, filename, **options):
        """Fit the source I data vs. DEC with a Gaussian +
           linear baseline"""
        mid = len(self.I_data) / 2
        amp_guess = self.I_data[mid] - self.I_data[0]
        center_guess = self.DEC[mid]
         # width in dec
        sigma_guess = np.abs(self.DEC[mid+3] - center_guess)
        slope_guess = (self.I_data[-1] - self.I_data[0])/(self.DEC[-1] -
                                                          self.DEC[0])
        y_int_guess = self.I_data[0] - slope_guess*self.DEC[0]
        guess_p = [amp_guess, center_guess, sigma_guess, y_int_guess,
                   slope_guess]
        #guess_p = [amp_guess, center_guess, sigma_guess, y_int_guess,
        #           slope_guess, slope_guess, slope_guess]
        sigma = [options["sigma"]]*len(self.DEC)
        try:
            fit_p, covar = curve_fit(gauss_and_line,  self.DEC,
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
                residuals = self.I_data - gauss_and_line(self.DEC,*fit_p)
                if (np.abs(self.e_fit_p[0]/self.fit_p[0])<options["amp_req"] and
                    np.abs(self.e_fit_p[2]/self.fit_p[2])<options["width_req"]):
                    self.good_fit = True
                    # determine center properties by finding closest point
                    # to center
                    center_point = np.abs(self.DEC - self.fit_p[1]).argmin()
                    self.center_RA = self.RA[center_point]
                    self.center_DEC = self.DEC[center_point]
                    self.I_baselined = (self.I_data -
                                        (self.fit_p[3] +
                                        self.fit_p[4]*self.DEC))
                    self.center_I = self.I_baselined[center_point]
                else:
                    self.good_fit = False
                # for plotting
                if options["file_verbose"]:
                    fit_x = np.linspace(self.DEC[0], self.DEC[-1], 100)
                    fit_y = gauss_and_line(fit_x, *fit_p)
                    make_plots.source_plot(self.DEC, self.I_data, residuals,
                                           fit_x, fit_y, filename)
        except RuntimeError:
            if options["verbose"]:
                print("Log: A fit did not converge.")
            self.good_fit = False
            

def gauss_and_line(x, *p):
    amp, center, sigma, y_int, slope = p
    return y_int + slope*x + amp*np.exp(-(x-center)**2/(2.*sigma**2))
    #amp,center,sigma,coeff0,coeff1,coeff2,coeff3 = p
    #return (coeff0 + coeff1*x + coeff2*x**2. + coeff3*x**3. +
    #        amp*np.exp(-(x-center)**2/(2.*sigma**2)))

if __name__ == "__main__":
    sys.exit("Error: module not mean to be run from top level")
