import numpy                as np
import matplotlib.pyplot    as plt
import os, os.path
import math             # per cross-correlazione 
from sklearn.metrics import mean_squared_error

plt.rcParams["font.family"] =   "sans-serif"
plt.rcParams['font.serif'] =    ['Helvetica']
plt.rcParams['font.size'] =     26

def preprocessing_bnn(data, sim_time_samples):

    '''reshape the peak train'''
    timestamps = []   
    neuron = data[:,1]
    for k in np.unique(neuron):
        index = np.where(neuron == k)[0]
        time_idx = np.where(data[index,0] < sim_time_samples)[0]    # to remove spikes associated to a sim_time bigger than the true sim_time (a kind of bug)
        timestamps.append(np.unique(data[index,0][time_idx]))       # to order and remove double spike in the same time (even though the csv is already sorted, there are some spikes repeated more than once without the cronological order)

    return timestamps

def preprocessing_snn(data, nneurons, sim_time_samples):

    '''reshape the peak train'''
    timestamps = []
    neuron = data[:,1]                                              
    for k in range(nneurons):
        index = np.where(neuron == k)[0]                            # remove automatically those rows where the neuron id is NaN
        time_idx = np.where(data[index,0] < sim_time_samples)[0]    # to remove spikes associated to a sim_time bigger than the true sim_time (a kind of bug)
                                                                    # and remove also those those rows in which only the time is NaN    
        timestamps.append(np.unique(data[index,0][time_idx]))       # to order and remove double spike in the same time (even though the csv is already sorted, there are some spikes repeated more than once without the cronological order) 
                                                                     
    return timestamps

def rasterplot(trials, rec_time, end=float('NaN'), start=float('NaN'), nneurons = 'all', fs = 1000, NN_type = 'NN', path = './', ext = ['svg'], fname = "RasterPlot"):

    '''raster plot'''
    fig, ax = plt.subplots(len(trials), layout = 'tight', figsize = (15, 5*len(trials)), squeeze=False)
    COLOR_DOTS = '#408A8E'                                      # Set colors of dots
    label = [NN_type + str(i+1) for i in range(len(trials))]
    for t in range(len(trials)):
        trial = trials[t]
        timestamps = []
        if nneurons == 'all':
            ntrial = len(trial)
        else:
            ntrial = nneurons
        for n in range(ntrial):
            if np.isnan(start) and np.isnan(end):
                neuron = trial[n]
            elif np.isnan(start) and not np.isnan(end):
                time_e = end*fs
                time_idx = np.where(trial[n] <= time_e)[0]
                neuron = trial[n][time_idx]
            elif not np.isnan(start) and np.isnan(end):
                time_s = start*fs
                time_idx = np.where(trial[n] >= time_s)[0]
                neuron = trial[n][time_idx]
            else:
                time_s = start*fs
                time_e = end*fs
                time_idx_e = np.where(trial[n] <= time_e)[0]
                time_idx = np.where(trial[n][time_idx_e] >= time_s)[0]
                neuron = trial[n][time_idx_e][time_idx]
            neuron_idx = np.ones(len(neuron))*n
            timestamp = np.array([neuron, neuron_idx])
            timestamps.append(timestamp.transpose())
        timestamps = np.concatenate(timestamps)
        x = timestamps[:,0]*1000/fs                             # in ms
        y = timestamps[:,1]
        ax[t][0].scatter(x, y, s=6, marker="|", color=COLOR_DOTS, linewidths=1)
        ax[t][0].set_xlabel('Time (ms)')
        ax[t][0].set_ylabel('Neuron index')
        ax[t][0].set_title(label[t]+' Rasterplot')
        if np.isnan(start) and np.isnan(end):
            ax[t][0].set_xlim([0, rec_time*1000])
        elif np.isnan(start) and not np.isnan(end):
            ax[t][0].set_xlim([0, end*1000])
        elif not np.isnan(start) and np.isnan(end):
            ax[t][0].set_xlim([start*1000, rec_time*1000])
        else:
            ax[t][0].set_xlim([start*1000, end*1000])

    plt.tight_layout()
    for t in ext:
        plt.savefig(os.path.join(path, fname + "." + t), format = t, bbox_inches = "tight")
    # plt.show()
    plt.close(fig)

    return

def correlation_coefficient(trial, snn_fs, rec_time, cc_type = 'binary signal', bin_time = 0.2):
    
    '''calculate two types of correlation coefficient'''
    print('correlation coefficient')
    snn_sim_time_samples = snn_fs*rec_time
    results=[]

    if cc_type == 'point process':
        for i in range(len(trial)-1):
            for j in range(i+1,len(trial)):
                x = trial[i]
                y = trial[j]
                Nx = len(x)
                Ny = len(y)
                r_xy = (snn_sim_time_samples*len(np.where(np.in1d(x,y))[0])-Nx*Ny)/(math.sqrt(Nx*(snn_sim_time_samples-Nx))*math.sqrt(Ny*(snn_sim_time_samples-Ny)))
                results.append([r_xy, i, j])

    elif cc_type == 'binary signal':
        nbin = int(rec_time/bin_time) 
        bin_samples = snn_fs*bin_time # in samples
        binary_sig = []
        for neuron in trial:
            bins_value = np.unique([int(spike/bin_samples) for spike in neuron])
            binary_sig.append(bins_value)
        for i in range(len(binary_sig)-1):
            for j in range(i+1,len(binary_sig)):
                x = binary_sig[i]
                y = binary_sig[j]
                Nx = len(x)
                Ny = len(y)
                if (math.sqrt(Nx*(nbin-Nx))*math.sqrt(Ny*(nbin-Ny))) == 0:
                    if (nbin*len(np.where(np.in1d(x,y))[0])-Nx*Ny)>0:
                        r_xy = 1
                    else:
                        r_xy = -1
                else:
                    r_xy = (nbin*len(np.where(np.in1d(x,y))[0])-Nx*Ny)/(math.sqrt(Nx*(nbin-Nx))*math.sqrt(Ny*(nbin-Ny)))
                results.append([r_xy, i, j])

    return results 
   
def plot_correlation_coefficient(CC_values, NN_type = 'NN', cc_bin_time = 0, path = './', ext = ['svg'], fname = "CorrelationHist"):

    fig, ax = plt.subplots(len(CC_values), layout = 'tight', figsize = (15, 5*len(CC_values)), squeeze=False)
    label = [NN_type + str(i+1) for i in range(len(CC_values))]
    for idx in range(len(CC_values)):
        trial = CC_values[idx]
        ax[idx][0].hist([cc[0] for cc in trial],np.linspace(-1,1,41))
        ax[idx][0].set_xlabel('Correlation')
        ax[idx][0].set_ylabel('Frequency')
        ax[idx][0].set_title(label[idx] + ' Correlation ' + str(int(cc_bin_time*1000)) + 'ms')
        ax[idx][0].set_xlim([-1,1])

    plt.tight_layout()
    for t in ext:
        plt.savefig(os.path.join(path, fname + "." + t), format = t, bbox_inches = "tight")
    # plt.show()
    plt.close(fig)

    return

def plot_box_linHist_correlation_coefficient(snn_CC_values, bnn_CC_values, cc_bin_time = 0, apx_line = 'median', path = './', ext = ['svg'], fname = "BoxLineCorrelation"):
    
    nfig = 2
    fig, ax = plt.subplots(nfig, layout = 'tight', figsize = (2*11,nfig*5))

    snn_label = ['SNN'+str(i+1) for i in range(len(snn_CC_values))]
    label = [*snn_label, *['BNN(RFA)']]

    # apx_line = 'median'            # <EDITABLE> Type of central line and variation: "median" "mean"
    lin_binEdges = np.linspace(-1,1,41)
    lin_binCenters = 0.5*(lin_binEdges[1:]+lin_binEdges[:-1])

    for itrial in range(len(snn_CC_values)):
        trial = snn_CC_values[itrial]
        Counts, _ = np.histogram([cc[0] for cc in trial],lin_binEdges)
        if sum(Counts)>0:
            normCounts = Counts/sum(Counts)
        else:
            normCounts = np.zeros(len(Counts))
        ax[0].plot(lin_binCenters,normCounts, label = label[itrial])
    bnn_isi_counts = []
    for itrial in range(len(bnn_CC_values)):
        trial = bnn_CC_values[itrial]
        Counts, _ = np.histogram([cc[0] for cc in trial],lin_binEdges)
        if sum(Counts)>0:
            normCounts = Counts/sum(Counts)
        else:
            normCounts = np.zeros(len(Counts))
        bnn_isi_counts.append(normCounts)
    mid = []
    inf = []
    sup = []
    if apx_line == 'median':
        for ibin in range(len(lin_binCenters)):
            bin_values = [trial[ibin] for trial in bnn_isi_counts]
            mid.append(np.median(bin_values))
            inf.append(np.percentile(bin_values, 25))
            sup.append(np.percentile(bin_values, 75))
            suffx = ' median + 25-75perc'
    elif apx_line == 'mean':
       for ibin in range(len(lin_binCenters)):
            bin_values = [trial[ibin] for trial in bnn_isi_counts]
            mid.append(np.mean(bin_values))
            inf.append(np.mean(bin_values)-2*np.std(bin_values))
            sup.append(np.mean(bin_values)+2*np.std(bin_values)) 
            suffx = ' mean +- 2std'
    ax[0].plot(lin_binCenters, mid, label = label[-1]+suffx)
    bcolors = [line.get_color() for line in ax[0].get_lines()]
    ax[0].fill_between(lin_binCenters, inf, sup, alpha=0.25, antialiased=True, color = bcolors[-1], linewidth=0)
    ax[0].set_xlabel('Correlation')
    ax[0].set_ylabel('Probability')
    ax[0].set_title("Pearson's Coefficient " + str(int(cc_bin_time*1000)) + 'ms')
    ax[0].legend(fontsize=10)

    snn_mean_val = []
    bnn_mean_val = []
    if apx_line == 'median':
        for trial in snn_CC_values:
            snn_mean_val.append(np.median([cc[0] for cc in trial]))
        for trial in bnn_CC_values:
            bnn_mean_val.append(np.median([cc[0] for cc in trial]))
        suffx = 'median '
    elif apx_line == 'mean':
        for trial in snn_CC_values:
            snn_mean_val.append(np.mean([cc[0] for cc in trial]))
        for trial in bnn_CC_values:
            bnn_mean_val.append(np.mean([cc[0] for cc in trial]))
        suffx = 'mean '
    bxp = ax[1].boxplot([*snn_mean_val, bnn_mean_val], patch_artist=True, medianprops=dict(color='k'), vert=True, meanline = True, showfliers= False, labels = label)
    ax[1].set_ylabel('Correlation')
    ax[1].set_title(suffx + "Pearson's Coefficient " + str(int(cc_bin_time*1000)) + 'ms')
    ax[1].grid(which = "both", axis = 'y')
    ax[1].minorticks_on()
    for patch, color in zip(bxp['boxes'], bcolors):
        patch.set_facecolor(color)

    plt.tight_layout()
    for t in ext:
        plt.savefig(os.path.join(path, fname + "." + t), format = t, bbox_inches = "tight")
    # plt.show()
    plt.close(fig)

    return snn_mean_val, bnn_mean_val

def spike_analysis(timestamps, fs, sim_time_samples=float('NaN'), spikeThCV = 10):

    '''calculate the mean firing rate and the number of spikes in each neuron signals'''
    mfr = []
    spikecounts = []
    for neuron in timestamps:
        spikecounts.append(len(neuron))
        if not np.isnan(sim_time_samples):
            mfr.append(len(neuron)/(sim_time_samples/fs))               # in [spikes/s]
    non_zeros_mfr = [i for i in mfr if i>0]

    '''calculate the isi of neuron signals and the absolute and relative number of neurons with no spike'''
    number_no_spikes_neurons = 0
    single_neur_isi = []
    for neuron in timestamps:
        if len(neuron) > 1 :
            single_neur_isi.append((neuron[1:]-neuron[:-1])*1000/fs)    # in ms
        if len(neuron)==0:
            number_no_spikes_neurons += 1
    overall_isi = np.concatenate(single_neur_isi, axis=0)               # in ms
    number_no_spikes_neurons_perc = number_no_spikes_neurons/len(timestamps)

    '''calculate the coefficient of variation for each single neuron and create a distribution'''
    nn_mean = []
    nn_std = []
    nn_cv = []
    for neuron in single_neur_isi:
        if len(neuron)>=spikeThCV: # se neurone ha almeno 11 spike -> e quindi 10 intervalli  
            std = np.std(neuron)/fs
            mean = np.mean(neuron)/fs
            nn_mean.append(mean)
            nn_std.append(std)
            nn_cv.append(std/mean)

    return mfr, non_zeros_mfr, spikecounts, single_neur_isi, overall_isi, number_no_spikes_neurons, number_no_spikes_neurons_perc, [nn_mean, nn_std, nn_cv]

def burst_analysis(timestamps, fs, sim_time_samples=float('NaN')):

    '''String method algorithm for the Burst Detection:
    calculate timestamps list of the first and last spike for all bursts and other parameters
    '''
    maxISI = 100                                                        # in ms
    minintraburstspikes = 5 
    burst_event_total = []
    end_burst_total = []
    intraburstspikes = []
    mbr = []
    burstcounts = []
    single_neur_burstlengths = []
    for neuron in timestamps :
        if len(neuron)>0:

            neuron = np.array(neuron)
            fake_spike=neuron[-1]+(maxISI*fs/1000)+1                        # in samples
            neuron = np.append(neuron, fake_spike)
    
            delta_time_spike = (neuron[1:] - neuron[:-1])*1000/fs           # in ms
            temp_mask_detection = delta_time_spike > maxISI                 # Change burst focusing when time delta >= 100 ms
            temp_mask_detection = np.append(True, temp_mask_detection)
            temp_time_burst_events = neuron[temp_mask_detection]
    
            burst_event_pos = np.where(np.in1d(neuron,temp_time_burst_events))[0]
            number_inburst_spike = burst_event_pos[1:] - burst_event_pos[:-1]
            mask_detection = number_inburst_spike >= minintraburstspikes    # Change the number of spikes in the burst >= 5
            mask_detection = np.append(mask_detection, False)
            time_burst_events = neuron[temp_mask_detection][mask_detection] # in samples
    
            idx_end_burst = np.where(np.in1d(neuron,time_burst_events))[0] + number_inburst_spike[mask_detection[:-1]] - 1
            time_end_burst = neuron[idx_end_burst]*1000/fs                  # in ms
    
            burst_event_total.append(time_burst_events*1000/fs)             # in ms
            end_burst_total.append(time_end_burst)                          # in ms
            intraburstspikes.append(number_inburst_spike[mask_detection[:-1]])
            if not np.isnan(sim_time_samples):
                mbr.append(len(time_burst_events)/(sim_time_samples/fs/60)) # in bursts/min
            burstcounts.append(len(time_burst_events))
            single_neur_burstlengths.append(time_end_burst-(time_burst_events*1000/fs)) # in ms
    non_zeros_mbr = [i for i in mbr if i>0]
    overall_burstlengths = np.concatenate(single_neur_burstlengths, axis=0)

    '''calculate the ibi of neuron signals'''
    number_no_bursts_neurons = 0
    single_neur_ibi = []
    for k in range(len(burstcounts)):
        if burstcounts[k] > 1:
            single_neur_ibi.append((burst_event_total[k][1:] - end_burst_total[k][:-1])) # in ms
        if burstcounts[k] == 0:
            number_no_bursts_neurons += 1
    if len(single_neur_ibi)>0:
        overall_ibi = np.concatenate(single_neur_ibi, axis=0)
    else:
        overall_ibi = []
    number_no_bursts_neurons_perc = number_no_bursts_neurons/len(timestamps)

    '''calculate Burstiness Index'''
    bin_width = 1000     # in ms (should be 1000 ms if we want to divide the recording in 1s-bins)
    factor = 0.15       # 0.15
    bin_width_samples = bin_width*fs/1000
    nbins = sim_time_samples/bin_width_samples
    nspikes_x_bin = []
    for b in range(int(nbins)):
        idx_inf = b*bin_width_samples
        idx_sup = (b+1)*bin_width_samples
        spikes_count = 0
        for neuron in timestamps:
            sup_lim = np.where(neuron < idx_sup)[0]
            both_lim = np.where(neuron[sup_lim] >= idx_inf)[0]
            spikes_count += len(both_lim)
        nspikes_x_bin.append(spikes_count)

    nspikes_x_bin_sorted = np.array(sorted(nspikes_x_bin, reverse=True))
    f15_bins = nspikes_x_bin_sorted[range(round(factor*nbins))]
    f15 = sum(f15_bins)/sum(nspikes_x_bin_sorted)
    BurtinessIndex = (f15-factor)/(1-factor)

    return burst_event_total, end_burst_total, intraburstspikes, mbr, non_zeros_mbr, burstcounts, single_neur_burstlengths, overall_burstlengths, single_neur_ibi, overall_ibi, number_no_bursts_neurons, number_no_bursts_neurons_perc, BurtinessIndex

def plot_spike_analysis(snn_spike_statistics, bnn_spike_statistics, bnn_spike_statistics_for_single_rats, apx_line = 'median', scale = 'log', xLimLin = 100, xBinLin = 50, path = './', ext = ['svg'], fname = "Spike_Analysis"):

    nfig = 8
    fig, ax = plt.subplots(nfig, layout = 'tight', figsize = (2*11,nfig*5))

    snn_label = ['SNN'+str(i+1) for i in range(len(snn_spike_statistics))]
    label = [*snn_label, *['BNN(RFA)']]

    if scale == 'log':
        tau = np.logspace(np.log10(min([min(i) for i in [*[nn[4] for nn in snn_spike_statistics], bnn_spike_statistics[4]] if len(i)>0])),np.log10(max([max(i) for i in [*[nn[4] for nn in snn_spike_statistics], bnn_spike_statistics[4]] if len(i)>0])), 30)
    else:
        tau = np.linspace(0,xLimLin,xBinLin)

    _, _, patches = ax[0].hist([*[nn[4] for nn in snn_spike_statistics], bnn_spike_statistics[4]], tau, edgecolor =  'k'  , stacked=False, label = label)
    ax[0].set_xlabel('ISI [ms]')
    ax[0].set_ylabel('Freq [spikes/bin]')
    if scale == 'log':
        ax[0].set_xscale('log')
    ax[0].legend(fontsize=10)
    ax[0].set_title('Inter-Spike Interval Histogram')
    bcolors = []
    for patch in patches:
        bcolors.append(patch[0].get_facecolor())

    bxp = ax[1].boxplot([*[nn[4] for nn in snn_spike_statistics], bnn_spike_statistics[4]], patch_artist=True, medianprops=dict(color='k'), vert=True, meanline = True, showfliers= False, labels = label)
    ax[1].set_ylabel('ISI [ms]')
    ax[1].set_title('Inter-Spike Interval')
    ax[1].grid(which = "both", axis = 'y')
    ax[1].minorticks_on()
    if scale == 'log':
        ax[1].set_yscale('log')
    for patch, color in zip(bxp['boxes'], bcolors):
        patch.set_facecolor(color)

    gridspec = ax[0].get_subplotspec().get_gridspec()
    ax[2].remove()
    subfig = fig.add_subfigure(gridspec[2])
    axs = subfig.subplots(1, len(snn_spike_statistics)+1)
    labels = 'off', 'on'
    colors = ['red', 'chartreuse']
    for i in range(len(snn_spike_statistics)):
        axs[i].pie([snn_spike_statistics[i][6], 1-snn_spike_statistics[i][6]], colors=colors, autopct='%1.1f%%', textprops={'size': 'smaller'})
        axs[i].set_title(snn_label[i], fontsize=22, y = 0.92)
        # axs[i].legend(patches, labels, loc="upper left", fontsize="10")
    patches, _, _ = axs[-1].pie([bnn_spike_statistics[6], 1-bnn_spike_statistics[6]], colors=colors, autopct='%1.1f%%', textprops={'size': 'smaller'})
    axs[-1].set_title('BNN(RFA)', fontsize=22, y = 0.92)
    axs[-1].legend(patches, labels, loc="lower right", fontsize="10")
    subfig.suptitle('Neurons spiking activity', fontsize=25, y = 1.05)

    ax[3].boxplot([*[nn[1] for nn in snn_spike_statistics], bnn_spike_statistics[1]], patch_artist=True, boxprops=dict(facecolor=colors[1]), medianprops=dict(color='k'), vert=True, meanline = True, showfliers= False, labels = label)
    ax[3].set_ylabel('MFR [spikes/s]')
    ax[3].set_title('Mean Firing Rate')
    ax[3].grid(which = "both", axis = 'y')
    ax[3].minorticks_on()
    if scale == 'log':
        ax[3].set_yscale('log')

    if scale == 'log':
        lin_binEdges = np.logspace(np.log10(min([min(i) for i in [*[nn[4] for nn in snn_spike_statistics], bnn_spike_statistics[4]] if len(i)>0])),np.log10(max([max(i) for i in [*[nn[4] for nn in snn_spike_statistics], bnn_spike_statistics[4]] if len(i)>0])), 30)
        log_binEdges = np.log10(lin_binEdges)
        log_binCenters = 0.5*(log_binEdges[1:]+log_binEdges[:-1])
        lin_binCenters = 10**log_binCenters
    else:
        lin_binEdges = np.linspace(0,xLimLin,xBinLin) # <EDIT> intervallo in sui fare l'isi... non si può considerare tutto perche altrimenti non si vede l'inizio del grafico
        lin_binCenters = 0.5*(lin_binEdges[1:]+lin_binEdges[:-1])

    snn_isi_counts = []
    for itrial in range(len(snn_spike_statistics)):
        all_isi = snn_spike_statistics[itrial][4]
        Counts, _ = np.histogram(all_isi,lin_binEdges)
        if sum(Counts)>0:
            normCounts = Counts/sum(Counts)
        else:
            normCounts = np.zeros(len(Counts))
        snn_isi_counts.append(normCounts)
        ax[4].plot(lin_binCenters,normCounts, label = label[itrial])
    
    bnn_isi_counts = []
    for trial in bnn_spike_statistics_for_single_rats:
        all_isi = trial[4]
        Counts, _ = np.histogram(all_isi,lin_binEdges)
        if sum(Counts)>0:
            normCounts = Counts/sum(Counts)
        else:
            normCounts = np.zeros(len(Counts))
        bnn_isi_counts.append(normCounts)
    mid = []
    inf = []
    sup = []
    if apx_line == 'median':
        for ibin in range(len(lin_binCenters)):
            bin_values = [trial[ibin] for trial in bnn_isi_counts]
            mid.append(np.median(bin_values))
            inf.append(np.percentile(bin_values, 25))
            sup.append(np.percentile(bin_values, 75))
            suffx = ' median + 25-75perc'
    elif apx_line == 'mean':
       for ibin in range(len(lin_binCenters)):
            bin_values = [trial[ibin] for trial in bnn_isi_counts]
            mid.append(np.mean(bin_values))
            inf.append(np.mean(bin_values)-2*np.std(bin_values))
            sup.append(np.mean(bin_values)+2*np.std(bin_values)) 
            suffx = ' mean +- 2std'
    ax[4].plot(lin_binCenters, mid, label = label[-1]+suffx)
    ax[4].fill_between(lin_binCenters, inf, sup, alpha=0.25, antialiased=True, color = bcolors[-1], linewidth=0)
    ax[4].set_xlabel('ISI [ms]')
    ax[4].set_ylabel('Probability')
    if scale == 'log':
        ax[4].set_xscale('log')
    ax[4].set_title('Inter-Spike Interval Histogram')
    ax[4].legend(fontsize=10)

    rmse_vector = []
    for snn in snn_isi_counts:
        rmse = mean_squared_error(mid, snn, squared=False)
        rmse_vector.append(rmse)

    snn_mean_val = []
    bnn_mean_val = []
    if apx_line == 'median':
        for itrial in range(len(snn_spike_statistics)):
            snn_mean_val.append(np.median(snn_spike_statistics[itrial][4]))
        for itrial in range(len(bnn_spike_statistics_for_single_rats)):
            bnn_mean_val.append(np.median(bnn_spike_statistics_for_single_rats[itrial][4]))
        suffx = 'median '
    elif apx_line == 'mean':
        for itrial in range(len(snn_spike_statistics)):
            snn_mean_val.append(np.mean(snn_spike_statistics[itrial][4]))
        for itrial in range(len(bnn_spike_statistics_for_single_rats)):
            bnn_mean_val.append(np.mean(bnn_spike_statistics_for_single_rats[itrial][4]))
        suffx = 'mean '
    bxp = ax[5].boxplot([*snn_mean_val, bnn_mean_val], patch_artist=True, medianprops=dict(color='k'), vert=True, meanline = True, showfliers= False, labels = label)
    ax[5].set_ylabel('ISI [ms]')
    ax[5].set_title(suffx + "Inter-Spike Interval")
    ax[5].grid(which = "both", axis = 'y')
    ax[5].minorticks_on()
    if scale == 'log':
        ax[5].set_yscale('log')
    for patch, color in zip(bxp['boxes'], bcolors):
        patch.set_facecolor(color)

    snn_cv = [isnn[7][2] for isnn in snn_spike_statistics]
    bnn_cv = [ibnn[7][2] for ibnn in bnn_spike_statistics_for_single_rats]
    lin_binEdges = np.linspace(0,max([max(nn) for nn in [*snn_cv, *bnn_cv]]),30) # <EDIT> intervallo in sui fare l'isi... non si può considerare tutto perche altrimenti non si vede l'inizio del grafico
    lin_binCenters = 0.5*(lin_binEdges[1:]+lin_binEdges[:-1])
    
    for i in range(len(snn_cv)):
        Counts, _ = np.histogram(snn_cv[i],lin_binEdges)
        normCounts = Counts/sum(Counts)
        ax[6].plot(lin_binCenters,normCounts, label = label[i])
    bnn_isi_counts = []
    for itrial in bnn_cv:
        Counts, _ = np.histogram(itrial,lin_binEdges)
        normCounts = Counts/sum(Counts)
        bnn_isi_counts.append(normCounts)
    
    mid = []
    inf = []
    sup = []
    if apx_line == 'median':
        for ibin in range(len(lin_binCenters)):
            bin_values = [trial[ibin] for trial in bnn_isi_counts]
            mid.append(np.median(bin_values))
            inf.append(np.percentile(bin_values, 25))
            sup.append(np.percentile(bin_values, 75))
            suffx = ' median + 25-75perc'
    elif apx_line == 'mean':
       for ibin in range(len(lin_binCenters)):
            bin_values = [trial[ibin] for trial in bnn_isi_counts]
            mid.append(np.mean(bin_values))
            inf.append(np.mean(bin_values)-2*np.std(bin_values))
            sup.append(np.mean(bin_values)+2*np.std(bin_values)) 
            suffx = ' mean +- 2std'
    ax[6].plot(lin_binCenters, mid, 'black', label = label[-1]+suffx)
    ax[6].fill_between(lin_binCenters, inf, sup, alpha=0.25, antialiased=True, color = 'black', linewidth=0)
    ax[6].legend(fontsize=10)
    ax[6].set_title('Coefficient of Variation')
    ax[6].set_xlabel('CV')
    ax[6].set_ylabel('Probability')

    snn_mean = np.concatenate([isnn[7][0] for isnn in snn_spike_statistics])
    snn_std = np.concatenate([isnn[7][1] for isnn in snn_spike_statistics])
    bnn_mean = np.concatenate([ibnn[7][0] for ibnn in bnn_spike_statistics_for_single_rats])
    bnn_std = np.concatenate([ibnn[7][1] for ibnn in bnn_spike_statistics_for_single_rats])
    
    ax[7].scatter(snn_mean, snn_std,s=1, color ='r', label = 'allSNNs')
    ax[7].scatter(bnn_mean, bnn_std,s=1, color ='b', label = 'allRATs')
    maxval = max([max(i) for i in [snn_mean, snn_std, bnn_mean, bnn_std]])
    minval = min([min(i) for i in [snn_mean, snn_std, bnn_mean, bnn_std]])
    ax[7].plot([minval,maxval],[minval,maxval], color = 'black', linestyle='--', linewidth=1)
    ax[7].set_xlim([minval, maxval])
    ax[7].set_ylim([minval, maxval])
    ax[7].set_xscale('log')
    ax[7].set_yscale('log')
    ax[7].legend(fontsize=10)
    ax[7].set_xlabel('mean')
    ax[7].set_ylabel('std.dev.')

    plt.tight_layout()
    for t in ext:
        plt.savefig(os.path.join(path, fname + "." + t), format = t, bbox_inches = "tight")
    # plt.show()
    plt.close(fig)

    return snn_mean_val, bnn_mean_val, rmse_vector

def plot_burst_analysis(snn_burst_statistics, bnn_burst_statistics, bnn_burst_statistics_for_single_rats, apx_line = 'median', scale = 'log', xLimLin = 1000, xBinLin = 50, path = './', ext = ['svg'], fname = "Burst_Analysis"):

    nfig = 11
    fig, ax = plt.subplots(nfig, layout = 'tight', figsize = (2*11,nfig*5))

    snn_label = ['SNN'+str(i+1) for i in range(len(snn_burst_statistics))]
    label = [*snn_label, *['BNN(RFA)']]

    if scale == 'log':
        tau = np.logspace(np.log10(min([min(i) for i in [*[nn[9] for nn in snn_burst_statistics], bnn_burst_statistics[9]] if len(i)>0])),np.log10(max([max(i) for i in [*[nn[9] for nn in snn_burst_statistics], bnn_burst_statistics[9]] if len(i)>0])), 30)
    else:
        tau = np.linspace(0,xLimLin,xBinLin)

    _, _, patches = ax[0].hist([*[nn[9] for nn in snn_burst_statistics], bnn_burst_statistics[9]], tau, edgecolor = 'k' , stacked=False, label = label)
    ax[0].set_xlabel('IBI [ms]')
    ax[0].set_ylabel('Freq [bursts/bin]')
    if scale == 'log':
        ax[0].set_xscale('log')
    ax[0].legend(fontsize=10)
    ax[0].set_title('Inter-Burst Interval Histogram')
    bcolors = []
    for patch in patches:
        bcolors.append(patch[0].get_facecolor())

    bxp = ax[1].boxplot([*[nn[9] for nn in snn_burst_statistics], bnn_burst_statistics[9]], patch_artist=True, medianprops=dict(color='k'), vert=True, meanline = True, showfliers= False, labels = label)
    ax[1].set_ylabel('IBI [ms]')
    ax[1].set_title('Inter-Burst Interval')
    ax[1].grid(which = "both", axis = 'y')
    ax[1].minorticks_on()
    if scale == 'log':
        ax[1].set_yscale('log')
    for patch, color in zip(bxp['boxes'], bcolors):
        patch.set_facecolor(color)

    gridspec = ax[0].get_subplotspec().get_gridspec()
    ax[2].remove()
    subfig = fig.add_subfigure(gridspec[2])
    axs = subfig.subplots(1, len(snn_burst_statistics)+1)
    labels = 'off', 'on'
    colors = ['red', 'chartreuse']
    for i in range(len(snn_burst_statistics)):
        axs[i].pie([snn_burst_statistics[i][11], 1-snn_burst_statistics[i][11]], colors=colors, autopct='%1.1f%%', textprops={'size': 'smaller'})
        axs[i].set_title(snn_label[i], fontsize=22, y = 0.92)
        # axs[i].legend(patches, labels, loc="upper left", fontsize="10")
    patches, _, _ = axs[-1].pie([bnn_burst_statistics[11], 1-bnn_burst_statistics[11]], colors=colors, autopct='%1.1f%%', textprops={'size': 'smaller'})
    axs[-1].set_title('BNN(RFA)', fontsize=22, y = 0.92)
    axs[-1].legend(patches, labels, loc="lower right", fontsize="10")
    subfig.suptitle('Neurons bursting activity', fontsize=25, y = 1.05)

    ax[3].boxplot([*[nn[4] for nn in snn_burst_statistics], bnn_burst_statistics[4]], patch_artist=True, boxprops=dict(facecolor=colors[1]), medianprops=dict(color='k'), vert=True, meanline = True, showfliers= False, labels = label)
    ax[3].set_ylabel('MBR [bursts/min]')
    ax[3].set_title('Mean Bursting Rate')
    ax[3].grid(which = "both", axis = 'y')
    ax[3].minorticks_on()
    if scale == 'log':
        ax[3].set_yscale('log')

    if scale == 'log':
        timelength_burst = np.logspace(np.log10(min([min(i) for i in [*[nn[7] for nn in snn_burst_statistics], bnn_burst_statistics[7]] if len(i)>0])),np.log10(max([max(i) for i in [*[nn[7] for nn in snn_burst_statistics], bnn_burst_statistics[7]] if len(i)>0])), 30)
    else:
        timelength_burst = np.linspace(0,xLimLin,xBinLin)

    _, _, patches = ax[4].hist([*[nn[7] for nn in snn_burst_statistics], bnn_burst_statistics[7]], timelength_burst, edgecolor = 'k' , stacked=False, label = label)
    ax[4].set_xlabel('Burstlength [ms]')
    ax[4].set_ylabel('Freq [bursts/bin]')
    if scale == 'log':
        ax[4].set_xscale('log')
    ax[4].legend(fontsize=10)
    ax[4].set_title('Burstlength Histogram')
    bcolors = []
    for patch in patches:
        bcolors.append(patch[0].get_facecolor())

    bxp = ax[5].boxplot([*[nn[7] for nn in snn_burst_statistics], bnn_burst_statistics[7]], patch_artist=True, medianprops=dict(color='k'), vert=True, meanline = True, showfliers= False, labels = label)
    ax[5].set_ylabel('Burstlength [ms]')
    ax[5].set_title('Burstlength')
    ax[5].grid(which = "both", axis = 'y')
    ax[5].minorticks_on()
    if scale == 'log':
        ax[5].set_yscale('log')
    for patch, color in zip(bxp['boxes'], bcolors):
        patch.set_facecolor(color)

    bxp = ax[6].boxplot([*[nn[12] for nn in snn_burst_statistics], [nn[12] for nn in bnn_burst_statistics_for_single_rats]], patch_artist=True, medianprops=dict(color='k'), vert=True, meanline = True, showfliers= False, labels = label)
    ax[6].set_ylabel('BI')
    ax[6].set_title('Burstiness Index')
    ax[6].grid(which = "both", axis = 'y')
    ax[6].minorticks_on()
    if scale == 'log':
        ax[6].set_yscale('log')
    for patch, color in zip(bxp['boxes'], bcolors):
        patch.set_facecolor(color)

    if scale == 'log':
        lin_binEdges = np.logspace(np.log10(min([min(i) for i in [*[nn[9] for nn in snn_burst_statistics], bnn_burst_statistics[9]] if len(i)>0])),np.log10(max([max(i) for i in [*[nn[9] for nn in snn_burst_statistics], bnn_burst_statistics[9]] if len(i)>0])), 30)
        log_binEdges = np.log10(lin_binEdges)
        log_binCenters = 0.5*(log_binEdges[1:]+log_binEdges[:-1])
        lin_binCenters = 10**log_binCenters
    else:
        lin_binEdges = np.linspace(0,xLimLin,xBinLin) # <EDIT> intervallo in sui fare l'isi... non si può considerare tutto perche altrimenti non si vede l'inizio del grafico
        lin_binCenters = 0.5*(lin_binEdges[1:]+lin_binEdges[:-1])

    for itrial in range(len(snn_burst_statistics)):
        all_isi = snn_burst_statistics[itrial][9]
        Counts, _ = np.histogram(all_isi,lin_binEdges)
        if sum(Counts)>0:
            normCounts = Counts/sum(Counts)
        else:
            normCounts = np.zeros(len(Counts))
        ax[7].plot(lin_binCenters,normCounts, label = label[itrial])
    bnn_isi_counts = []
    for trial in bnn_burst_statistics_for_single_rats:
        all_isi = trial[9]
        Counts, _ = np.histogram(all_isi,lin_binEdges)
        if sum(Counts)>0:
            normCounts = Counts/sum(Counts)
        else:
            normCounts = np.zeros(len(Counts))
        bnn_isi_counts.append(normCounts)
    mid = []
    inf = []
    sup = []
    if apx_line == 'median':
        for ibin in range(len(lin_binCenters)):
            bin_values = [trial[ibin] for trial in bnn_isi_counts]
            mid.append(np.median(bin_values))
            inf.append(np.percentile(bin_values, 25))
            sup.append(np.percentile(bin_values, 75))
            suffx = ' median + 25-75perc'
    elif apx_line == 'mean':
       for ibin in range(len(lin_binCenters)):
            bin_values = [trial[ibin] for trial in bnn_isi_counts]
            mid.append(np.mean(bin_values))
            inf.append(np.mean(bin_values)-2*np.std(bin_values))
            sup.append(np.mean(bin_values)+2*np.std(bin_values)) 
            suffx = ' mean +- 2std'
    ax[7].plot(lin_binCenters, mid, label = label[-1]+suffx)
    ax[7].fill_between(lin_binCenters, inf, sup, alpha=0.25, antialiased=True, color = bcolors[-1], linewidth=0)
    ax[7].set_xlabel('IBI [ms]')
    ax[7].set_ylabel('Probability')
    if scale == 'log':
        ax[7].set_xscale('log')
    ax[7].legend(fontsize=10)
    ax[7].set_title('Inter-Burst Interval Histogram')

    snn_ibi_val = []
    bnn_ibi_val = []
    if apx_line == 'median':
        for itrial in range(len(snn_burst_statistics)):
            snn_ibi_val.append(np.median(snn_burst_statistics[itrial][9]))
        for itrial in range(len(bnn_burst_statistics_for_single_rats)):
            bnn_ibi_val.append(np.median(bnn_burst_statistics_for_single_rats[itrial][9]))
        suffx = 'median '
    elif apx_line == 'mean':
        for itrial in range(len(snn_burst_statistics)):
            snn_ibi_val.append(np.mean(snn_burst_statistics[itrial][9]))
        for itrial in range(len(bnn_burst_statistics_for_single_rats)):
            bnn_ibi_val.append(np.mean(bnn_burst_statistics_for_single_rats[itrial][9]))
        suffx = 'mean '
    bxp = ax[8].boxplot([*snn_ibi_val, bnn_ibi_val], patch_artist=True, medianprops=dict(color='k'), vert=True, meanline = True, showfliers= False, labels = label)
    ax[8].set_ylabel('IBI [ms]')
    ax[8].set_title(suffx + "Inter-Burst Interval")
    ax[8].grid(which = "both", axis = 'y')
    ax[8].minorticks_on()
    if scale == 'log':
        ax[8].set_yscale('log')
    for patch, color in zip(bxp['boxes'], bcolors):
        patch.set_facecolor(color)

    if scale == 'log':
        lin_binEdges = np.logspace(np.log10(min([min(i) for i in [*[nn[7] for nn in snn_burst_statistics], bnn_burst_statistics[7]] if len(i)>0])),np.log10(max([max(i) for i in [*[nn[7] for nn in snn_burst_statistics], bnn_burst_statistics[7]] if len(i)>0])), 30)
        log_binEdges = np.log10(lin_binEdges)
        log_binCenters = 0.5*(log_binEdges[1:]+log_binEdges[:-1])
        lin_binCenters = 10**log_binCenters
    else:
        lin_binEdges = np.linspace(0,xLimLin,xBinLin) # <EDIT> intervallo in sui fare l'isi... non si può considerare tutto perche altrimenti non si vede l'inizio del grafico
        lin_binCenters = 0.5*(lin_binEdges[1:]+lin_binEdges[:-1])

    for itrial in range(len(snn_burst_statistics)):
        all_isi = snn_burst_statistics[itrial][7]
        Counts, _ = np.histogram(all_isi,lin_binEdges)
        if sum(Counts)>0:
            normCounts = Counts/sum(Counts)
        else:
            normCounts = np.zeros(len(Counts))
        ax[9].plot(lin_binCenters,normCounts, label = label[itrial])
    bnn_isi_counts = []
    for trial in bnn_burst_statistics_for_single_rats:
        all_isi = trial[7]
        Counts, _ = np.histogram(all_isi,lin_binEdges)
        if sum(Counts)>0:
            normCounts = Counts/sum(Counts)
        else:
            normCounts = np.zeros(len(Counts))
        bnn_isi_counts.append(normCounts)
    mid = []
    inf = []
    sup = []
    if apx_line == 'median':
        for ibin in range(len(lin_binCenters)):
            bin_values = [trial[ibin] for trial in bnn_isi_counts]
            mid.append(np.median(bin_values))
            inf.append(np.percentile(bin_values, 25))
            sup.append(np.percentile(bin_values, 75))
            suffx = ' median + 25-75perc'
    elif apx_line == 'mean':
       for ibin in range(len(lin_binCenters)):
            bin_values = [trial[ibin] for trial in bnn_isi_counts]
            mid.append(np.mean(bin_values))
            inf.append(np.mean(bin_values)-2*np.std(bin_values))
            sup.append(np.mean(bin_values)+2*np.std(bin_values)) 
            suffx = ' mean +- 2std'
    ax[9].plot(lin_binCenters, mid, label = label[-1]+suffx)
    ax[9].fill_between(lin_binCenters, inf, sup, alpha=0.25, antialiased=True, color = bcolors[-1], linewidth=0)
    ax[9].set_xlabel('Burstlength [ms]')
    ax[9].set_ylabel('Probability')
    if scale == 'log':
        ax[9].set_xscale('log')
    ax[9].legend(fontsize=10)
    ax[9].set_title('Burstlength Histogram')

    snn_bl_val = []
    bnn_bl_val = []
    if apx_line == 'median':
        for itrial in range(len(snn_burst_statistics)):
            snn_bl_val.append(np.median(snn_burst_statistics[itrial][7]))
        for itrial in range(len(bnn_burst_statistics_for_single_rats)):
            bnn_bl_val.append(np.median(bnn_burst_statistics_for_single_rats[itrial][7]))
        suffx = 'median '
    elif apx_line == 'mean':
        for itrial in range(len(snn_burst_statistics)):
            snn_bl_val.append(np.mean(snn_burst_statistics[itrial][7]))
        for itrial in range(len(bnn_burst_statistics_for_single_rats)):
            bnn_bl_val.append(np.mean(bnn_burst_statistics_for_single_rats[itrial][7]))
        suffx = 'mean '
    bxp = ax[10].boxplot([*snn_bl_val, bnn_bl_val], patch_artist=True, medianprops=dict(color='k'), vert=True, meanline = True, showfliers= False, labels = label)
    ax[10].set_ylabel('Burstlength [ms]')
    ax[10].set_title(suffx + 'Burstlength')
    ax[10].grid(which = "both", axis = 'y')
    ax[10].minorticks_on()
    if scale == 'log':
        ax[10].set_yscale('log')
    for patch, color in zip(bxp['boxes'], bcolors):
        patch.set_facecolor(color)

    plt.tight_layout()
    for t in ext:
        plt.savefig(os.path.join(path, fname + "." + t), format = t, bbox_inches = "tight")
    # plt.show()
    plt.close(fig)

    return snn_ibi_val, bnn_ibi_val, snn_bl_val, bnn_bl_val

def plot_single_neuron_isi(nn_spike_statistics, NN_type = 'NN', yLim = 0.15, apx_line = 'median', scale = 'log', xLimLin = 100, xBinLin = 50, path = './', ext = ['svg'], fname = "SingleNeuronISI"):

    '''rappresentazione in scala logaritmica -- linee+fascia'''
    nrow = len(nn_spike_statistics)
    ncol = 2
    fig, ax = plt.subplots(nrow, ncol, layout = 'tight', figsize = (15*ncol,nrow*5), squeeze=False)
    label = [NN_type + str(i+1) for i in range(nrow)]

    if scale == 'log':
        lin_binEdges = np.logspace(np.log10(min([min(nn[4]) for nn in nn_spike_statistics if len(nn[4])>0])),np.log10(max([max(nn[4]) for nn in nn_spike_statistics if len(nn[4])>0])), 30)
        log_binEdges = np.log10(lin_binEdges)
        log_binCenters = 0.5*(log_binEdges[1:]+log_binEdges[:-1])
        lin_binCenters = 10**log_binCenters
    else:
        lin_binEdges = np.linspace(0,xLimLin,xBinLin) # <EDIT> intervallo in sui fare l'isi... non si può considerare tutto perche altrimenti non si vede l'inizio del grafico
        lin_binCenters = 0.5*(lin_binEdges[1:]+lin_binEdges[:-1])
    
    bnn_isi_counts = []
    for itrial in range(len(nn_spike_statistics)):
        trialcount = []
        for neuron in nn_spike_statistics[itrial][3]:
            Counts, _ = np.histogram(neuron,lin_binEdges)
            if sum(Counts)>0:
                normCounts = Counts/sum(Counts)
            else:
                normCounts = np.zeros(len(Counts))
            trialcount.append(normCounts)
            ax[itrial][0].plot(lin_binCenters,normCounts)
        if scale == 'log':
            ax[itrial][0].set_xscale('log')
        ax[itrial][0].set_ylim([0,yLim])
        ax[itrial][0].set_xlabel('ISI [ms]')
        ax[itrial][0].set_ylabel('Probability')
        ax[itrial][0].set_title(label[itrial] + " neurons' ISI")
        bnn_isi_counts.append(trialcount)

    mid = []
    inf = []
    sup = []
    if apx_line == 'median':
        for trial in bnn_isi_counts:
            mid_trial = []
            inf_trial = []
            sup_trial = []
            for ibin in range(len(lin_binCenters)):
                bin_values = [neurons[ibin] for neurons in trial]
                mid_trial.append(np.median(bin_values))
                inf_trial.append(np.percentile(bin_values, 25))
                sup_trial.append(np.percentile(bin_values, 75))
                suffx = ' median '
            mid.append(mid_trial)
            inf.append(inf_trial)
            sup.append(sup_trial)
    elif apx_line == 'mean':
        for trial in bnn_isi_counts:
            mid_trial = []
            inf_trial = []
            sup_trial = []
            for ibin in range(len(lin_binCenters)):
                bin_values = [neurons[ibin] for neurons in trial]
                mid_trial.append(np.mean(bin_values))
                inf_trial.append(np.mean(bin_values)-2*np.std(bin_values))
                sup_trial.append(np.mean(bin_values)+2*np.std(bin_values)) 
                suffx = ' mean '
            mid.append(mid_trial)
            inf.append(inf_trial)
            sup.append(sup_trial)

    for itrial in range(len(mid)):
        ax[itrial][1].plot(lin_binCenters, mid[itrial])
        ax[itrial][1].fill_between(lin_binCenters, inf[itrial], sup[itrial], alpha=0.25, antialiased=True, linewidth=0)
        if scale == 'log':
            ax[itrial][1].set_xscale('log')
        ax[itrial][1].set_ylim([0,yLim])
        ax[itrial][1].set_xlabel('ISI [ms]')
        ax[itrial][1].set_ylabel('Probability')
        ax[itrial][1].set_title(label[itrial] + suffx + ' ISIH')
    
    # fig.suptitle(suffx + "Inter-Spike Interval")

    plt.tight_layout()
    for t in ext:
        plt.savefig(os.path.join(path, fname + "." + t), format = t, bbox_inches = "tight")
    # plt.show()
    plt.close(fig)

    return