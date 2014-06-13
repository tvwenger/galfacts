"""
find_sources.py
Find sources in GALFACTS transient search
03 June 2014 - Trey Wenger - creation
12 June 2014 - Trey Wenger - fixed smoothing convolution normalization
                             bug in beam.py
"""
vers = "v1.0.1"

import sys
import os
import argparse
import numpy as np
import beam

def main(**options):
    """Main script for finding GALFACTS sources"""
    print("find_sources.py {0}".format(vers))
    for b in options["beams"]:
        if options["verbose"]:
            print("Log: Starting beam {0} analysis.".format(b))
        this_beam = beam.Beam(b, **options)
        this_beam.find_sources()
    if options["verbose"]:
        print("Log: Done!")

if __name__ == "__main__":
    parser=argparse.ArgumentParser(
        description="Search GALFACTS for sources.",
        prog='find_sources.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--version', action='version',
                        version='%(prog)s '+vers)
    required=parser.add_argument_group('required arguments:')
    required.add_argument('--field',type=str,
                          help="field to analyze",
                          required=True)
    required.add_argument('--date',type=int,
                          help="date to analyze",
                          required=True)
    semi_opt=parser.add_argument_group('arguments set to defaults:')
    semi_opt.add_argument('--data_filepath',type=str,
                          help='path where data are saved',
                          default="../data")
    semi_opt.add_argument('--results_filepath',type=str,
                          help='path where results are saved',
                          default="../results")
    semi_opt.add_argument('--beams',type=int,nargs="+",
                          help='beams to analyze',
                          default=[0,1,2,3,4,5,6])
    semi_opt.add_argument('--ra_corr',type=float,nargs=7,
                          help='RA beam location correction',
                          metavar=("BEAM0","BEAM1","BEAM2","BEAM3",
                                   "BEAM4","BEAM5","BEAM6"),
                          default=[0.,2.7417,5.4833,2.7417,-2.7417,
                                   -5.4833,-2.7417])
    semi_opt.add_argument('--num_channels',type=int,
                          help='number of channels in observation',
                          default=2048)
    semi_opt.add_argument('--bin_width',type=float,
                          help='width of analysis bins in MHz',
                          default=5.)
    semi_opt.add_argument('--band_width',type=float,
                          help='width of band in MHz',
                          default=172.5)
    semi_opt.add_argument('--rfi_con_width',type=int,
                          help='convolution half-width for RFI detection '+
                          'in channels',
                          default=1)
    semi_opt.add_argument('--smooth_con_width',type=int,
                          help='convolution half-width for smoothing '+
                          'in time points',
                          default=28)
    semi_opt.add_argument('--source_con_width',type=int,
                          help='convolution width for source '+
                          'detection in time points',
                          default=24)
    semi_opt.add_argument('--edge_buff_chan',type=int,
                          help='channels to cut from convolution '+
                          'edge',
                          default=125)
    semi_opt.add_argument('--edge_buff_time',type=int,
                          help='time points to cut from convolution '+
                          'edge',
                          default=52)
    semi_opt.add_argument('--num_intervals',type=int,
                          help='number of intervals for calculating '+
                          'RFI mask',
                          default=10)
    semi_opt.add_argument('--rfi_mask',type=float,
                          help='sigma for RFI cut',
                          default=6.)
    semi_opt.add_argument('--source_mask',type=float,
                          help='sigma (SNR) for source cut',
                          default=8.)
    semi_opt.add_argument('--sigma',type=float,
                          help='theoretical noise level in Kelvin',
                          default=0.017)
    semi_opt.add_argument('--num_source_points',type=int,
                          help='number of points to fit around '+
                          'each source peak',
                          default=8)
    semi_opt.add_argument('--point_sep',type=int,
                          help='number of points to skip in fiiting '+
                          'source (skips sidelobes)',
                          default=25)
    semi_opt.add_argument('--num_outer_points',type=int,
                          help='number of points to fit to get '+
                          'baseline level',
                          default=5)
    semi_opt.add_argument('--ast_offset',type=float,
                          help='fraction offset correction to AST '+
                          'coordinates and time',
                          default=0.5)
    semi_opt.add_argument('--amp_req',type=float,
                          help='Source fit e_amplitude/amplitude requirement',
                          default=0.1)
    semi_opt.add_argument('--width_req',type=float,
                          help='Source fit e_width/width requirement',
                          default=0.1)
#    semi_opt.add_argument('--dec',type=float,nargs=2,
#                          metavar=('LOWER','UPPER'),
#                          help="analyze only this declination range, "+
#                          "inclusively, in degrees",
#                          default=[-90.,90.])
    optional=parser.add_argument_group('other optional arguments:')
    optional.add_argument('--exclude_channels',type=int,nargs='+',
                          help="channels to exclude")
    optional.add_argument('-v','--verbose',help="verbose analysis",
                          action="store_true")
    optional.add_argument('-f','--file_verbose',
                          help="make lots of intermediate files",
                          action="store_true")
    args = vars(parser.parse_args())
    main(**args)
