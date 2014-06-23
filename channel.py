"""
channel.py
Channel object for GALFACTS transient search
02 June 2014 - Trey Wenger - creation
"""
import os
import sys
import numpy as np

class Channel(object):
    """Channel object for GALFACTS transient search"""
    def __init__(self, chan_num, beam_num, **options):
        """Initialize the channel object"""
        self.chan_file = "{0}/{1}/{2}/beam{3}/fluxtime{4:04d}.dat".\
          format(options["data_filepath"],
                 options["field"],
                 options["date"],
                 beam_num,
                 chan_num)
        ra,dec,ast,I,Q,U,V = np.loadtxt(self.chan_file,unpack=True)
        self.num_points = len(ra)
        self.error = (not os.path.isfile(self.chan_file))

    def average(self):
        """Return the average Stokes for this channel"""
        ra,dec,ast,I,Q,U,V = np.loadtxt(self.chan_file,unpack=True)
        self.num_points = len(ra)
        return (np.mean(I), np.mean(Q), np.mean(U), np.mean(V))

    def add_points(self, Iarr, Qarr, Uarr, Varr):
        """Add these channel's points to the running I, Q, U, V total
           for each timestamp"""
        ra,dec,ast,I,Q,U,V = np.loadtxt(self.chan_file,unpack=True)
        return (Iarr + I, Qarr + Q, Uarr + U, Varr + V)

    def get_coordinates(self):
        """Get the AST, RA, and DEC for this channel"""
        ra,dec,ast,I,Q,U,V = np.loadtxt(self.chan_file,unpack=True)
        return ra, dec, ast

if __name__ == "__main__":
    sys.exit("Error: module not meant to be run at top level.")
