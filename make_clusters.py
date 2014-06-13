"""
make_clusters.py
Take output from find_sources.py and determine source positions
that are related using mean shift algorithm. Fit the data from
the related sources with a 2D Gaussian + baseplane.
06 June 2014 - Trey Wenger - Creation
"""
vers = "v1.0"

import sys
import os
import argparse
import numpy as np
import source
import make_plots as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
import cluster

def main(**options):
    """Main script for clustering algorithm"""
    print("make_clusters.py {0}".format(vers))
    # Analysis will be done in each bin individually
    my_dir = "{0}/{1}/{2}/beam{3}".\
      format(options["source_filepath"],
             options["field"],
             options["dates"][0],
             options["beams"][0])
    bins = [o for o in os.listdir(my_dir)
            if os.path.isdir(os.path.join(my_dir,o))]
    for this_bin in bins:
        # First, gather up the center positions for each source point
        if options["verbose"]:
            print("Log: Working on {0}".format(this_bin))
        sources = []
        for date in options["dates"]:
            if options["verbose"]:
                print("Log: Working on date {0}".format(date))
            for beam in options["beams"]:
                if options["verbose"]:
                    print("Log: Working on beam {0}".format(beam))
                my_in_dir = "{0}/{1}/{2}/beam{3}/{4}".format(
                    options["source_filepath"],
                    options["field"],date,beam,this_bin)
                if not os.path.isdir(my_in_dir):
                    continue
                data = np.load(my_in_dir+"/sources.npz")
                num_good = 0
                for s in data["sources"]:
                    if s.good_fit:
                        sources.append(s)
                        num_good += 1
                if options["verbose"]:
                    print("Log: found {0} good sources.".\
                          format(num_good))
        if options["verbose"]:
            print("Log: found {0} total good sources in this bin."\
                  .format(len(sources)))
        if len(sources) == 0:
            continue
        RA = np.float64(np.array([s.center_RA for s in sources]))
        DEC = np.float64(np.array([s.center_DEC for s in sources]))
        I_data = [s.center_I for s in sources]
        my_out_dir = "{0}/{1}/{2}".\
          format(options["cluster_filepath"],
                 options["field"],this_bin)
        if not os.path.isdir(my_out_dir):
            os.makedirs(my_out_dir)
        if options["file_verbose"]:
            plt.field_plot(RA, DEC, I_data,
                           my_out_dir+"/all_sources.png")
        if options["verbose"]:
            print("Log: clustering members.")
        n_clusters = 0
        try:
            # estimate mean shift bandwidth
            # need to correct for cosine dec.
            X = np.array(zip(RA*np.cos(np.deg2rad(DEC)), DEC))
            bandwidth = estimate_bandwidth(X,
                                           quantile=options["quantile"],
                                           n_samples=len(X))
            if options["verbose"]:
                print("Log: found bandwidth {0}".format(bandwidth))
            ms = MeanShift(bandwidth=bandwidth,bin_seeding=True)
            ms.fit(X)
            labels = ms.labels_
            centers = ms.cluster_centers_
            labels_unique = np.unique(labels)
            n_clusters = len(labels_unique)
            if options["verbose"]:
                print("Log: found {0} clusters.".format(n_clusters))
            if options["file_verbose"]:
                plt.field_plot(RA, DEC, I_data,
                            my_out_dir+"/clustered.png",
                            labels=labels, centers=centers)
        except ValueError:
            if options["verbose"]:
                print("Log: Could not implement clustering!")
        if n_clusters > 0:
            # make plots and fit each cluster
            if options["verbose"]:
                print("Log: fitting clusters")
            clusters = []
            for clust in range(n_clusters):
                my_members = labels == clust
                my_sources = [sources[i] for i in
                              range(len(my_members)) if my_members[i]]
                # gather up date for sources in this cluster
                my_RA = np.array([])
                my_DEC = np.array([])
                my_AST = np.array([])
                my_I_data = np.array([])
                my_Q_data = np.array([])
                my_U_data = np.array([])
                my_V_data = np.array([])
                for src in my_sources:
                    my_RA = np.append(my_RA,src.RA)
                    my_DEC = np.append(my_DEC,src.DEC)
                    my_AST = np.append(my_AST,src.AST)
                    my_I_data = np.append(my_I_data,src.I_baselined)
                    my_Q_data = np.append(my_Q_data,src.Q_data)
                    my_U_data = np.append(my_U_data,src.U_data)
                    my_V_data = np.append(my_V_data,src.V_data)
                if options["file_verbose"]:
                    plt.field_plot(my_RA, my_DEC, my_I_data,
                                   my_out_dir+"/cluster{0:03d}.png".\
                                   format(clust))
                clusters.append(cluster.Cluster(my_RA, my_DEC, my_AST,
                                                my_I_data, my_Q_data,
                                                my_U_data, my_V_data))
            good_clusters = []
            bad_clusters = []
            for clust in range(len(clusters)):
                clusters[clust].fit(my_out_dir+"/cluster{0:03d}_fit.png".\
                                    format(clust),
                                    **options)
                if clusters[clust].good_fit:
                    good_clusters.append(clust)
                else:
                    bad_clusters.append(clust)
            if options["verbose"]:
                print("Log: Fit {0} good clusters.".format(len(good_clusters)))
                print("Log: Found {0} bad clusters.".format(len(bad_clusters)))
        
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
                          help="field to cluster",
                          required=True)
    required.add_argument('--dates',type=int,nargs="+",
                          help="dates to cluster",
                          required=True)
    semi_opt=parser.add_argument_group('arguments set to defaults:')
    semi_opt.add_argument('--source_filepath',type=str,
                          help='path where sources are saved',
                          default="../results")
    semi_opt.add_argument('--cluster_filepath',type=str,
                          help='path where cluster results will go',
                          default="../results/clusters")
    semi_opt.add_argument('--beams',type=int,nargs="+",
                          help='beams to cluster',
                          default=[0,1,2,3,4,5,6])
    semi_opt.add_argument('--quantile',type=float,
                          help='Quantile parameter for mean shift algorithm',
                          default=0.1)
    semi_opt.add_argument('--sigma',type=float,
                          help='theoretical noise level in Kelvin',
                          default=0.017)
    semi_opt.add_argument('--beam_width',type=float,
                          help='Telescope beamwidth, guess for cluster fit',
                          default=0.058)
    semi_opt.add_argument('--amp_req',type=float,
                          help='Source fit e_amplitude/amplitude requirement',
                          default=0.1)
    semi_opt.add_argument('--width_req',type=float,
                          help='Source fit e_width/width requirement',
                          default=0.1)
    optional=parser.add_argument_group('other optional arguments:')
    optional.add_argument('-v','--verbose',help="verbose analysis",
                          action="store_true")
    optional.add_argument('-f','--file_verbose',
                          help="make lots of intermediate files",
                          action="store_true")
    args = vars(parser.parse_args())
    main(**args)
