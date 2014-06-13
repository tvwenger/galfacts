"""
beam.py
Beam object for GALFACTS transient search
02 June 2014 - Trey Wenger - creation
12 June 2014 - Trey Wenger - Fixed smoothing convolution norm. problem
"""
import sys
import os
import numpy as np
import channel
import make_plots as plt
import source

class Beam(object):
    """Beam object for GALFACTS transient search"""
    def __init__(self, beam_num, **options):
        """Initialize the beam object"""
        self.beam_num = beam_num
        self.options = options
        self.channels = [channel.Channel(i, beam_num, **options) for
                         i in xrange(options["num_channels"])]
        # put error in channel zero. This is the Calgary average
        self.channels[0].error = True
        # put error in  ignored channels
        if options["exclude_channels"] != None:
            for c in options["exclude_channels"]:
                self.channels[c].error = True

    def find_sources(self):
        """Algorithm to detect sources for this beam"""
        # generate results directory
        results_dir = "{0}/{1}/{2}/beam{3}".\
          format(self.options["results_filepath"],
                 self.options["field"],
                 self.options["date"],
                 self.beam_num)
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        
        # Stokes averaged over time
        if self.options["verbose"]:
            print("Log: Averaging Stokes over time.")
        I_data = np.zeros(self.options["num_channels"])
        Q_data = np.zeros(self.options["num_channels"])
        U_data = np.zeros(self.options["num_channels"])
        V_data = np.zeros(self.options["num_channels"])
        for i in xrange(self.options["num_channels"]):
            if self.channels[i].error: continue
            I_data[i], Q_data[i], U_data[i], V_data[i] = \
              self.channels[i].average()
        if self.options["file_verbose"]:
            np.savez(results_dir+"/time_avg",
                     channel=range(self.options["num_channels"]),
                     I_data = I_data,
                     Q_data = Q_data,
                     U_data = U_data,
                     V_data = V_data)
            chans = range(0,self.options["num_channels"])
            plt.stokes_plot(chans, "Channel", I_data, Q_data, U_data,
                            V_data, results_dir+"/time_avg.png")
            
        # convolve time-avereraged data to detect RFI
        if self.options["verbose"]:
            print("Log: Performing RFI detection convolution.")
        con = np.zeros(2*self.options["rfi_con_width"] + 1)
        con[0] = -0.5
        con[self.options["rfi_con_width"]] = 1.0
        con[-1] = -0.5
        I_data = np.convolve(I_data,con,mode="same")
        Q_data = np.convolve(Q_data,con,mode="same")
        U_data = np.convolve(U_data,con,mode="same")
        V_data = np.convolve(V_data,con,mode="same")
        if self.options["file_verbose"]:
            np.savez(results_dir+"/rfi_conv_time_avg",
                     channel=range(self.options["num_channels"]),
                     I_data = I_data,
                     Q_data = Q_data,
                     U_data = U_data,
                     V_data = V_data)
            chans = range(0,self.options["num_channels"])
            plt.stokes_plot(chans, "Channel", I_data, Q_data, U_data,
                            V_data,
                            results_dir+"/rfi_conv_time_avg.png")

        # eliminated edge channels
        if self.options["verbose"]:
            print("Log: eliminating edge channels.")
        for i in xrange(self.options["edge_buff_chan"]):
            self.channels[i].error=True
            self.channels[-1-i].error=True
        # determine the minimum mean and std dev in our intervals
        interval_width = self.options["num_channels"]/\
          self.options["num_intervals"]
        for data in [I_data, Q_data, U_data, V_data]:
            means = np.array([np.nanmean(data[i:i+interval_width])
                              for i in
                              range(0,self.options["num_channels"],
                                    interval_width)])
            stddevs = np.array([np.nanstd(data[i:i+interval_width])
                                for i in
                                range(0,self.options["num_channels"],
                                      interval_width)])
            min_ind = stddevs[stddevs.nonzero()].argmin()
            min_stddev = stddevs[min_ind]
            min_mean = means[min_ind]
            # determine bad channels and flag them
            bad_chans = np.where(np.abs(data) > min_mean +\
                                 self.options["rfi_mask"]*min_stddev)[0]
            if self.options["verbose"]:
                print("Log: eliminated {0} channels with RFI".\
                      format(len(bad_chans)))
            for c in bad_chans:
                self.channels[c].error = True

        # recompute Stokes averaged over time
        if self.options["verbose"]:
            print("Log: Recomputing average Stokes over time.")
        I_data = np.zeros(self.options["num_channels"])
        Q_data = np.zeros(self.options["num_channels"])
        U_data = np.zeros(self.options["num_channels"])
        V_data = np.zeros(self.options["num_channels"])
        for i in xrange(self.options["num_channels"]):
            if self.channels[i].error: continue
            I_data[i], Q_data[i], U_data[i], V_data[i] = \
              self.channels[i].average()
        if self.options["file_verbose"]:
            np.savez(results_dir+"/clean_time_avg",
                     channel=range(self.options["num_channels"]),
                     I_data = I_data,
                     Q_data = Q_data,
                     U_data = U_data,
                     V_data = V_data)
            chans = range(0,self.options["num_channels"])
            plt.stokes_plot(chans, "Channel", I_data, Q_data, U_data,
                            V_data,
                            results_dir+"/clean_time_avg.png")
            
        # now, detect sources in each bin as well as over the full
        # bandpass
        num_bins = int(self.options["band_width"]/
                       self.options["bin_width"])+1
        chans_per_bin = int(self.options["num_channels"]/num_bins)
        for b in range(num_bins+1):
            if b == num_bins:
                b = 999
                start_chan = 0
                end_chan = self.options["num_channels"]
            else:
                start_chan = b*chans_per_bin
                end_chan = (b+1)*chans_per_bin
                if end_chan > self.options["num_channels"]:
                    end_chan = self.options["num_channels"]
            # check if we're already outside edge buffer
            if (end_chan < self.options["edge_buff_chan"] or
                start_chan > (self.options["num_channels"] -
                              self.options["edge_buff_chan"])):
                continue
            if self.options["verbose"]:
                print("Log: Analyzing bin {0}".format(b))
            # results directory for this bin
            bin_results_dir = results_dir+"/bin{0:03d}".format(b)
            if not os.path.isdir(bin_results_dir):
                os.makedirs(bin_results_dir)
            # Average over channels
            if self.options["verbose"]:
                print("Log: Averaging Stokes over channels in this bin")
            I_data = np.zeros(self.channels[0].num_points)
            Q_data = np.zeros(self.channels[0].num_points)
            U_data = np.zeros(self.channels[0].num_points)
            V_data = np.zeros(self.channels[0].num_points)
            num_good_points = 0.
            for c in xrange(start_chan,end_chan):
                if self.channels[c].error: continue
                I_data, Q_data, U_data, V_data = \
                  self.channels[c].add_points(I_data,Q_data,U_data,
                                              V_data)
                num_good_points += 1.
            if num_good_points == 0.: continue
            I_data /= num_good_points
            Q_data /= num_good_points
            U_data /= num_good_points
            V_data /= num_good_points
            if self.options["verbose"]:
                print("Log: Correcting coordinates.")
            RA, DEC, AST = get_coordinates(self.beam_num,
                                           **self.options)
            if self.options["file_verbose"]:
                np.savez(bin_results_dir+"/chan_avg",
                         chan_range=[start_chan,end_chan],
                         RA = RA,
                         DEC = DEC,
                         AST = AST,
                         I_data = I_data,
                         Q_data = Q_data,
                         U_data = U_data,
                         V_data = V_data)
                plt.stokes_plot(AST, "AST", I_data, Q_data, U_data,
                                V_data,
                                bin_results_dir+"/chan_avg.png")

            # smooth data
            if self.options["verbose"]:
                print("Log: Performing smoothing convolution.")
            angle = np.arange(2*self.options["smooth_con_width"]+1)
            angle = angle*10.*np.pi/(2.*self.options["smooth_con_width"])
            angle = angle - 5.*np.pi
            con = np.sin(angle)/angle
            con[self.options["smooth_con_width"]] = 1.
            I_data = np.convolve(I_data,con,mode="same")/np.sum(con)
            Q_data = np.convolve(Q_data,con,mode="same")/np.sum(con)
            U_data = np.convolve(U_data,con,mode="same")/np.sum(con)
            V_data = np.convolve(V_data,con,mode="same")/np.sum(con)
            # chop off edges after convolution
            if self.options["verbose"]:
                print("Log: Chopping off edges after convolution")
            RA = RA[self.options["edge_buff_time"]:
                    -self.options["edge_buff_time"]]
            DEC = DEC[self.options["edge_buff_time"]:
                      -self.options["edge_buff_time"]]
            AST = AST[self.options["edge_buff_time"]:
                      -self.options["edge_buff_time"]]
            I_data = I_data[self.options["edge_buff_time"]:
                            -self.options["edge_buff_time"]]
            Q_data = Q_data[self.options["edge_buff_time"]:
                            -self.options["edge_buff_time"]]
            U_data = U_data[self.options["edge_buff_time"]:
                            -self.options["edge_buff_time"]]
            V_data = V_data[self.options["edge_buff_time"]:
                            -self.options["edge_buff_time"]]
            if self.options["file_verbose"]:
                np.savez(bin_results_dir+"/smooth_chan_avg",
                         chan_range=[start_chan,end_chan],
                         RA = RA,
                         DEC = DEC,
                         AST = AST,
                         I_data = I_data,
                         Q_data = Q_data,
                         U_data = U_data,
                         V_data = V_data)
                plt.stokes_plot(AST, "AST", I_data, Q_data, U_data,
                                V_data,
                                bin_results_dir+"/smooth_chan_avg.png")
            
            # convolve for source detection
            if self.options["verbose"]:
                print("Log: Performing source detection convolution.")
            con = np.zeros(2*self.options["source_con_width"]+1)
            con[0] = -0.25
            con[self.options["source_con_width"]/2] = -0.25
            con[self.options["source_con_width"]] = 1.0
            con[3*self.options["source_con_width"]/2] = -0.25
            con[-1] = -0.25
            I_data_source = np.convolve(I_data,con,mode="same")
            if self.options["file_verbose"]:
                np.savez(bin_results_dir+"/source_chan_avg",
                         chan_range=[start_chan,end_chan],
                         RA = RA,
                         DEC = DEC,
                         AST = AST,
                         I_data = I_data_source)
                plt.single_stokes(AST, "AST", I_data_source,
                                  "Stokes I (K)",
                                  bin_results_dir+
                                  "/source_chan_avg.png")
            if self.options["verbose"]:
                print("Log: Locating sources.")
            source_points = np.where(I_data_source >
                                     self.options["source_mask"]*
                                     self.options["sigma"])[0]
            # storage for sources
            sources = []
            # i is the starting point for this source
            i=0
            while i < len(source_points):
                # j is the ending point for this source
                j = i+1
                # as long as [j] = [j-1] + 1, still on same source
                while (j < len(source_points) and
                       source_points[j] == source_points[j-1] + 1):
                    j += 1
                # get the necessary data for this source
                # first find max
                this_I_data = I_data_source[source_points[i]:
                                            source_points[j-1]]
                if len(this_I_data) == 0:
                    i = j
                    continue
                # max point in I_data_source array for this source
                max_point = this_I_data.argmax() + source_points[i]
                # now, get coords and data for fitting
                time_end = False
                base1_start = (max_point-
                               self.options["num_source_points"]-
                               self.options["point_sep"]-
                               self.options["num_outer_points"])
                if base1_start < 0: base1_start = 0
                base1_end = base1_start+self.options["num_outer_points"]
                if base1_end < 0: base1_end = 0
                source_start = max_point-self.options["num_source_points"]
                if source_start < 0: source_start = 0
                source_end = max_point+self.options["num_source_points"]+1
                if source_end > len(AST): source_end = len(AST)
                base2_start = (max_point+1+
                               self.options["num_source_points"]+
                               self.options["point_sep"])
                if base2_start > len(AST): base2_start = len(AST)
                base2_end = base2_start+self.options["num_outer_points"]
                if base2_end > len(AST): base2_end = len(AST)
                # source is near end of observation
                if base1_start < 0 or base2_end > len(AST):
                    time_end = True
                this_RA = RA[base1_start:base1_end]
                this_RA = np.append(this_RA,RA[source_start:source_end])
                this_RA = np.append(this_RA,RA[base2_start:base2_end])
                this_DEC = DEC[base1_start:base1_end]
                this_DEC = np.append(this_DEC,DEC[source_start:source_end])
                this_DEC = np.append(this_DEC,DEC[base2_start:base2_end])
                this_AST = AST[base1_start:base1_end]
                this_AST = np.append(this_AST,AST[source_start:source_end])
                this_AST = np.append(this_AST,AST[base2_start:base2_end])
                this_I_data = I_data[base1_start:base1_end]
                this_I_data = np.append(this_I_data,I_data[source_start:source_end])
                this_I_data = np.append(this_I_data,I_data[base2_start:base2_end])
                this_Q_data = Q_data[base1_start:base1_end]
                this_Q_data = np.append(this_Q_data,Q_data[source_start:source_end])
                this_Q_data = np.append(this_Q_data,Q_data[base2_start:base2_end])
                this_U_data = U_data[base1_start:base1_end]
                this_U_data = np.append(this_U_data,U_data[source_start:source_end])
                this_U_data = np.append(this_U_data,U_data[base2_start:base2_end])
                this_V_data = V_data[base1_start:base1_end]
                this_V_data = np.append(this_V_data,V_data[source_start:source_end])
                this_V_data = np.append(this_V_data,V_data[base2_start:base2_end])
                # sstart = max_point-25
                # send = max_point+25
                # if sstart < 0:
                #     time_end = True
                #     sstart = 0
                # if send > len(AST):
                #     time_end = True
                #     send = len(AST)
                # this_RA = RA[sstart:send]
                # this_DEC = DEC[sstart:send]
                # this_AST = AST[sstart:send]
                # this_I_data = I_data[sstart:send]
                # this_Q_data = Q_data[sstart:send]
                # this_U_data = U_data[sstart:send]
                # this_V_data = V_data[sstart:send]
                # check dec scan to see if we change direction
                # across source
                dec_end = False
                for k in range(len(this_DEC)-1):
                    if (np.sign(this_DEC[k+1]-this_DEC[k]) !=
                        np.sign(this_DEC[1]-this_DEC[0])):
                        dec_end = True
                        break
                # now, add it
                sources.append(source.Source(this_RA, this_DEC, this_AST,
                                             this_I_data, this_Q_data,
                                             this_U_data, this_V_data,
                                             time_end, dec_end))
                # start next search from where this one left off
                i = j
            if self.options["verbose"]:
                print("Log: Found {0} sources.".format(len(sources)))
                print("Log: Fitting good sources.")
            good_sources = []
            bad_sources = []
            for s in range(len(sources)):
                if sources[s].time_end or sources[s].dec_end:
                    bad_sources.append(s)
                else:
                    plt_filename = bin_results_dir+"/source{0:03d}".format(s)
                    sources[s].fit(plt_filename, **self.options)
                    if sources[s].good_fit:
                        good_sources.append(s)
                    else:
                        bad_sources.append(s)
            if self.options["verbose"]:
                print("Log: Fit {0} good sources.".format(len(good_sources)))
                print("Log: Found {0} bad sources.".format(len(bad_sources)))
            if self.options["file_verbose"]:
                with open(bin_results_dir+"/good_sources.txt","w") as f:
                    f.write("# SourceNum centerRA centerDEC peakI widthDEC\n")
                    f.write("# --------- deg      deg       K     deg\n")
                    for s in good_sources:
                        f.write("{0:03d} {1:.3f} {2:.3f} {3:.3f} {4:.3f}\n".\
                                format(s,sources[s].center_RA,
                                       sources[s].center_DEC,
                                       sources[s].center_I,
                                       sources[s].fit_p[2]))
                with open(bin_results_dir+"/bad_sources.txt","w") as f:
                    f.write("# SourceNum Reasons\n")
                    for s in bad_sources:
                        reasons = ""
                        if sources[s].dec_end:
                            reasons = reasons+"DecChange,"
                        if sources[s].time_end:
                            reasons = reasons+"EndOfScan,"
                        if not sources[s].good_fit:
                            reasons = reasons+"BadFit,"
                        f.write("{0:03d} {1}\n".format(s,reasons))
            np.savez(bin_results_dir+"/sources",
                     sources=sources)
            
def get_coordinates(beam_num, **options):
    """Return the AST, RA, and DEC for the specified beam after
       performing the necessary corrections"""
    # just use some random channel number, but make it different
    # for each beam in case we multithread this - we probably don't
    # want to open the same file in two places at once
    num = options["num_channels"] - beam_num - 1
    # first, get the RA and AST from beam 0 then apply corrections
    temp_chan = channel.Channel(num, 0, **options)
    RA, skipDEC, AST = temp_chan.get_coordinates()
    # correct RA and AST for AST offset
    for i in xrange(len(AST)-1):
        AST[i] = AST[i] + (AST[i+1] - AST[i])*options["ast_offset"]
        RA[i] = RA[i] + (RA[i+1] - RA[i])*options["ast_offset"]
    # now get the DEC for this beam
    temp_chan = channel.Channel(num, beam_num, **options)
    skipRA, DEC, skipAST = temp_chan.get_coordinates()
    # correct DEC for AST offset
    for i in xrange(len(DEC)-1):
        DEC[i] = DEC[i] + (DEC[i+1] - DEC[i])*options["ast_offset"]
    # correct RA for RA correction
    RA = RA + options["ra_corr"][beam_num]/(60. * np.cos(np.deg2rad(DEC)))
    return RA, DEC, AST
    
if __name__ == "__main__":
    sys.exit("Error: module not meant to be run at top level.")
