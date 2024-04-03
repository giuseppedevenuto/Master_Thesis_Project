from SnnBnnNSA              import *

def NSA(snn_category, bnn_category, rec_time, cc_type, cc_bin_time):
    '''SNN'''
    print('SNN data loading...')
    snn_nneurons = 1024
    snn_fs = 1000                           # Hz [samples/s]
    snn_sim_time_samples = rec_time*snn_fs  # in samples
    snn_fdir = './Thesis Data/snn'
    snn_fnames = [fname for fname in os.listdir(snn_fdir) if os.path.isfile(os.path.join(snn_fdir,fname)) and snn_category in fname] 
    snn_data = [np.genfromtxt(os.path.join(snn_fdir,ifname), skip_header=1, delimiter=';') for isnn in np.arange(len(snn_fnames)) for ifname in snn_fnames if 'SNN'+str(isnn+1)+'-' in ifname]
    print('SNN data analysis...')
    snn_timestamps = [preprocessing_snn(trial, snn_nneurons, snn_sim_time_samples) for trial in snn_data]
    snn_spike_statistics = [spike_analysis(timestamp, snn_fs, snn_sim_time_samples) for timestamp in snn_timestamps] 
    snn_burst_statistics = [burst_analysis(timestamp, snn_fs, snn_sim_time_samples) for timestamp in snn_timestamps]
    snn_CC_values = [correlation_coefficient(trial, snn_fs, rec_time, cc_type = cc_type, bin_time = cc_bin_time) for trial in snn_timestamps]

    '''BNN'''
    print('BNN data loading...')
    bnn_fs = 25000                          # Hz [samples/s]
    bnn_sim_time_samples = rec_time*bnn_fs  # in samples
    bnn_fdir = './Thesis Data/bnn'
    bnn_data = [np.genfromtxt(os.path.join(bnn_fdir,fname), skip_header=1, delimiter=',') for fname in os.listdir(bnn_fdir) if os.path.isfile(os.path.join(bnn_fdir,fname)) and bnn_category in fname]
    print('BNN data analysis...')
    bnn_timestamps = [preprocessing_bnn(rat, bnn_sim_time_samples) for rat in bnn_data]
    bnn_spike_statistics_for_single_rats = [spike_analysis(timestamp, bnn_fs, bnn_sim_time_samples) for timestamp in bnn_timestamps] 
    bnn_burst_statistics_for_single_rats = [burst_analysis(timestamp, bnn_fs, bnn_sim_time_samples) for timestamp in bnn_timestamps]
    bnn_CC_values = [correlation_coefficient(trial, bnn_fs, rec_time, cc_type = cc_type, bin_time = cc_bin_time) for trial in bnn_timestamps]

    bnn_timestamps_unified = [neuron for rat in bnn_timestamps for neuron in rat]
    bnn_spike_statistics = spike_analysis(bnn_timestamps_unified, bnn_fs, bnn_sim_time_samples)
    bnn_burst_statistics = burst_analysis(bnn_timestamps_unified, bnn_fs, bnn_sim_time_samples)


    '''Plots'''
    ext = ['svg','jpg']
    endTime_raster   = 90 # <EDIT> in seconds
    startTime_raster = 60 # <EDIT> in seconds
    if 'old' in snn_category:
        savedir = './Thesis Images/Old Topology Results/Old Topology'

        name_snn_raster = 'RasterOldSnn'
        name_snn_raster30 = 'RasterOldSnn30n'
        name_snn_correlation = 'CorrCoeffHistOldSnn'
        medianLogNameSnnSingleNeuronISI = 'SingleISIMedianLogOldSnn'
        meanLogNameSnnSingleNeuronISI = 'SingleISIMeanLogOldSnn'
        medianLinNameSnnSingleNeuronISI = 'SingleISIMedianLinOldSnn'
        meanLinNameSnnSingleNeuronISI = 'SingleISIMeanLinOldSnn'

        name_bnn_raster = 'RasterRat'
        name_bnn_raster30 = 'RasterRat30n'
        name_bnn_correlation = 'CorrCoeffHistRat'
        medianLogNameBnnSingleNeuronISI = 'SingleISIMedianLogRat'
        meanLogNameBnnSingleNeuronISI = 'SingleISIMeanLogRat'
        medianLinNameBnnSingleNeuronISI = 'SingleISIMedianLinRat'
        meanLinNameBnnSingleNeuronISI = 'SingleISIMeanLinRat'

        medianBoxLineCorrelation = 'CorrCoeffBoxLineMedianOld'
        medianLogNameSpike = 'SpikeResultMedianLogOld'
        medianLogNameBurst = 'BurstResultMedianLogOld'
        medianLinNameSpike = 'SpikeResultMedianLinOld'
        medianLinNameBurst = 'BurstResultMedianLinOld'

        meanBoxLineCorrelation = 'CorrCoeffBoxLineMeanOld'
        meanLogNameSpike = 'SpikeResultMeanLogOld'
        meanLogNameBurst = 'BurstResultMeanLogOld'
        meanLinNameSpike = 'SpikeResultMeanLinOld'
        meanLinNameBurst = 'BurstResultMeanLinOld'

    elif 'CONFIG' in snn_category:
        savedir = './Thesis Images/SW Topology Results/SW Topology'

        name_snn_raster = 'RasterNewSnn'
        name_snn_raster30 = 'RasterNewSnn30n'
        name_snn_correlation = 'CorrCoeffHistNewSnn'
        medianLogNameSnnSingleNeuronISI = 'SingleISIMedianLogNewSnn'
        meanLogNameSnnSingleNeuronISI = 'SingleISIMeanLogNewSnn'
        medianLinNameSnnSingleNeuronISI = 'SingleISIMedianLinNewSnn'
        meanLinNameSnnSingleNeuronISI = 'SingleISIMeanLinNewSnn'

        name_bnn_raster = 'RasterRat'
        name_bnn_raster30 = 'RasterRat30n'
        name_bnn_correlation = 'CorrCoeffHistRat'
        medianLogNameBnnSingleNeuronISI = 'SingleISIMedianLogRat'
        meanLogNameBnnSingleNeuronISI = 'SingleISIMeanLogRat'
        medianLinNameBnnSingleNeuronISI = 'SingleISIMedianLinRat'
        meanLinNameBnnSingleNeuronISI = 'SingleISIMeanLinRat'

        medianBoxLineCorrelation = 'CorrCoeffBoxLineMedianNew'
        medianLogNameSpike = 'SpikeResultMedianLogNew'
        medianLogNameBurst = 'BurstResultMedianLogNew'
        medianLinNameSpike = 'SpikeResultMedianLinNew'
        medianLinNameBurst = 'BurstResultMedianLinNew'

        meanBoxLineCorrelation = 'CorrCoeffBoxLineMeanNew'
        meanLogNameSpike = 'SpikeResultMeanLogNew'
        meanLogNameBurst = 'BurstResultMeanLogNew'
        meanLinNameSpike = 'SpikeResultMeanLinNew'
        meanLinNameBurst = 'BurstResultMeanLinNew'
      
    savedir = './Thesis Images/'+snn_category+' Results/'+snn_category+'1000BIbin-raster30sec-binsignal'+str(int(cc_bin_time*1000))+'ms' # <EDIT>
    os.makedirs(savedir)

    '''SNN DATA'''
    print('SNN data plotting...')
    rasterplot(snn_timestamps, rec_time = rec_time, end = endTime_raster, start = startTime_raster, nneurons = 'all', fs = snn_fs, NN_type = 'SNN', path = savedir, ext = ext, fname = name_snn_raster) 
    rasterplot(snn_timestamps, rec_time = rec_time, end = endTime_raster, start = startTime_raster, nneurons = 30, fs = snn_fs, NN_type = 'SNN', path = savedir, ext = ext, fname = name_snn_raster30) 
    plot_correlation_coefficient(snn_CC_values, 'SNN', cc_bin_time, path = savedir, ext = ext, fname = name_snn_correlation)
    # Log Median 
    plot_single_neuron_isi(snn_spike_statistics, 'SNN', yLim = 0.4, apx_line = 'median', scale = 'log', path = savedir, ext = ext, fname = medianLogNameSnnSingleNeuronISI)
    # # Log Mean
    # plot_single_neuron_isi(snn_spike_statistics, 'SNN', yLim = 0.4, apx_line = 'mean', scale = 'log', path = savedir, ext = ext, fname = meanLogNameSnnSingleNeuronISI)
    # Lin Median
    plot_single_neuron_isi(snn_spike_statistics, 'SNN', yLim = 0.7, apx_line = 'median', scale = 'lin', xLimLin = 100, xBinLin = 50, path = savedir, ext = ext, fname = medianLinNameSnnSingleNeuronISI)
    # # Lin Mean
    # plot_single_neuron_isi(snn_spike_statistics, 'SNN', yLim = 0.7, apx_line = 'mean', scale = 'lin', xLimLin = 100, xBinLin = 50, path = savedir, ext = ext, fname = meanLinNameSnnSingleNeuronISI)

    '''BNN DATA'''
    print('BNN data plotting...')
    rasterplot(bnn_timestamps, rec_time = rec_time, end = endTime_raster, start = startTime_raster, nneurons = 'all', fs = bnn_fs, NN_type = 'RAT', path = savedir, ext = ext, fname = name_bnn_raster) 
    rasterplot(bnn_timestamps, rec_time = rec_time, end = endTime_raster, start = startTime_raster, nneurons = 30, fs = bnn_fs, NN_type = 'RAT', path = savedir, ext = ext, fname = name_bnn_raster30) 
    plot_correlation_coefficient(bnn_CC_values, 'RAT', cc_bin_time, path = savedir, ext = ext, fname = name_bnn_correlation)
    # Log Median 
    plot_single_neuron_isi(bnn_spike_statistics_for_single_rats, 'RAT', yLim = 0.15, apx_line = 'median', scale = 'log', path = savedir, ext = ext, fname = medianLogNameBnnSingleNeuronISI)
    # # Log Mean
    # plot_single_neuron_isi(bnn_spike_statistics_for_single_rats, 'RAT', yLim = 0.15, apx_line = 'mean', scale = 'log', path = savedir, ext = ext, fname = meanLogNameBnnSingleNeuronISI)
    # Lin Median
    plot_single_neuron_isi(bnn_spike_statistics_for_single_rats, 'RAT', yLim = 0.6, apx_line = 'median', scale = 'lin', xLimLin = 100, xBinLin = 50, path = savedir, ext = ext, fname = medianLinNameBnnSingleNeuronISI)
    # # Lin Mean
    # plot_single_neuron_isi(bnn_spike_statistics_for_single_rats, 'RAT', yLim = 0.6, apx_line = 'mean', scale = 'lin', xLimLin = 100, xBinLin = 50, path = savedir, ext = ext, fname = meanLinNameBnnSingleNeuronISI)

    '''BOTH NN DATA'''
    print('Final plots...')
    snn_cc_val, bnn_cc_val = plot_box_linHist_correlation_coefficient(snn_CC_values, bnn_CC_values, cc_bin_time, apx_line = 'median', path = savedir, ext = ext, fname = medianBoxLineCorrelation)
    # Log Median
    snn_isi_val_log, bnn_isi_val_log, snn_rmse_vector_log = plot_spike_analysis(snn_spike_statistics, bnn_spike_statistics, bnn_spike_statistics_for_single_rats, apx_line = 'median', scale = 'log', path = savedir, ext = ext, fname = medianLogNameSpike) 
    snn_ibi_val_log, bnn_ibi_val_log, snn_bl_val_log, bnn_bl_val_log = plot_burst_analysis(snn_burst_statistics, bnn_burst_statistics, bnn_burst_statistics_for_single_rats, apx_line = 'median', scale = 'log', path = savedir, ext = ext, fname = medianLogNameBurst) 
    # Lin Median
    plot_spike_analysis(snn_spike_statistics, bnn_spike_statistics, bnn_spike_statistics_for_single_rats, apx_line = 'median', scale = 'lin', xLimLin = 100, xBinLin = 50, path = savedir, ext = ext, fname = medianLinNameSpike) 
    plot_burst_analysis(snn_burst_statistics, bnn_burst_statistics, bnn_burst_statistics_for_single_rats, apx_line = 'median', scale = 'lin', xLimLin = 1000, xBinLin = 50, path = savedir, ext = ext, fname = medianLinNameBurst) 

    # plot_box_linHist_correlation_coefficient(snn_CC_values, bnn_CC_values, cc_bin_time, apx_line = 'mean', path = savedir, ext = ext, fname = meanBoxLineCorrelation)
    # # Log Mean
    # plot_spike_analysis(snn_spike_statistics, bnn_spike_statistics, bnn_spike_statistics_for_single_rats, apx_line = 'mean', scale = 'log', path = savedir, ext = ext, fname = meanLogNameSpike) 
    # plot_burst_analysis(snn_burst_statistics, bnn_burst_statistics, bnn_burst_statistics_for_single_rats, apx_line = 'mean', scale = 'log', path = savedir, ext = ext, fname = meanLogNameBurst) 
    # # Lin Mean
    # plot_spike_analysis(snn_spike_statistics, bnn_spike_statistics, bnn_spike_statistics_for_single_rats, apx_line = 'mean', scale = 'lin', xLimLin = 100, xBinLin = 50, path = savedir, ext = ext, fname = meanLinNameSpike) 
    # plot_burst_analysis(snn_burst_statistics, bnn_burst_statistics, bnn_burst_statistics_for_single_rats, apx_line = 'mean', scale = 'lin', xLimLin = 1000, xBinLin = 50, path = savedir, ext = ext, fname = meanLinNameBurst) 

    print('Computation successfully accomplished!')

    return [snn_cc_val, bnn_cc_val], [snn_isi_val_log, bnn_isi_val_log], [snn_ibi_val_log, bnn_ibi_val_log], [snn_bl_val_log, bnn_bl_val_log], [[nn[12] for nn in snn_burst_statistics], [nn[12] for nn in bnn_burst_statistics_for_single_rats]], [np.concatenate([nn[4] for nn in snn_spike_statistics], axis=0), bnn_spike_statistics[4]], [np.concatenate([nn[1] for nn in snn_spike_statistics], axis=0), bnn_spike_statistics[1]], [np.concatenate([nn[9] for nn in snn_burst_statistics], axis=0), bnn_burst_statistics[9]], [np.concatenate([nn[4] for nn in snn_burst_statistics], axis=0), bnn_burst_statistics[4]], [np.concatenate([nn[7] for nn in snn_burst_statistics], axis=0), bnn_burst_statistics[7]], [np.concatenate([isnn[7][0] for isnn in snn_spike_statistics]),np.concatenate([ibnn[7][0] for ibnn in bnn_spike_statistics_for_single_rats])], [np.concatenate([isnn[7][1] for isnn in snn_spike_statistics]),np.concatenate([ibnn[7][1] for ibnn in bnn_spike_statistics_for_single_rats])], [snn_rmse_vector_log, [0]]

def plot_configs(pc, isi_median, ibi_median, bl_median, bi, isi, mfr, ibi, mbr, bl, cv_mean, cv_std, configsName = ['X'], scale = 'log', path = './', ext = ['svg'], fname = "ConfigsComparison"):

    nfig = 11
    fig, ax = plt.subplots(nfig, layout = 'tight', figsize = (len(pc)*3,nfig*5))

    snn_label = ['CONF'+str(i) for i in configsName]
    label = [*snn_label, *['BNN(RFA)']]

    ax[0].boxplot(pc, patch_artist=True, medianprops=dict(color='k'), vert=True, meanline = True, showfliers= False, labels = label)
    ax[0].set_ylabel('Correlation')
    ax[0].set_title("median Pearson's Coefficient")
    ax[0].grid(which = "both", axis = 'y')
    ax[0].minorticks_on()

    ax[1].boxplot(mfr, patch_artist=True, medianprops=dict(color='k'), vert=True, meanline = True, showfliers= False, labels = label)
    ax[1].set_ylabel('MFR [spikes/s]')
    ax[1].set_title("Mean Firing Rate")
    ax[1].grid(which = "both", axis = 'y')
    ax[1].minorticks_on()
    if scale == 'log':
        ax[1].set_yscale('log')

    ax[2].boxplot(isi, patch_artist=True, medianprops=dict(color='k'), vert=True, meanline = True, showfliers= False, labels = label)
    ax[2].set_ylabel('ISI [ms]')
    ax[2].set_title("Inter-Spike Interval")
    ax[2].grid(which = "both", axis = 'y')
    ax[2].minorticks_on()
    if scale == 'log':
        ax[2].set_yscale('log')

    ax[3].boxplot(isi_median, patch_artist=True, medianprops=dict(color='k'), vert=True, meanline = True, showfliers= False, labels = label)
    ax[3].set_ylabel('ISI [ms]')
    ax[3].set_title("median Inter-Spike Interval")
    ax[3].grid(which = "both", axis = 'y')
    ax[3].minorticks_on()
    if scale == 'log':
        ax[3].set_yscale('log')

    ax[4].boxplot(mbr, patch_artist=True, medianprops=dict(color='k'), vert=True, meanline = True, showfliers= False, labels = label)
    ax[4].set_ylabel('MBR [bursts/min]')
    ax[4].set_title("Mean Bursting Rate")
    ax[4].grid(which = "both", axis = 'y')
    ax[4].minorticks_on()
    if scale == 'log':
        ax[4].set_yscale('log')

    ax[5].boxplot(ibi, patch_artist=True, medianprops=dict(color='k'), vert=True, meanline = True, showfliers= False, labels = label)
    ax[5].set_ylabel('IBI [ms]')
    ax[5].set_title("Inter-Burst Interval")
    ax[5].grid(which = "both", axis = 'y')
    ax[5].minorticks_on()
    if scale == 'log':
        ax[5].set_yscale('log')
    
    ax[6].boxplot(ibi_median, patch_artist=True, medianprops=dict(color='k'), vert=True, meanline = True, showfliers= False, labels = label)
    ax[6].set_ylabel('IBI [ms]')
    ax[6].set_title("median Inter-Burst Interval")
    ax[6].grid(which = "both", axis = 'y')
    ax[6].minorticks_on()
    if scale == 'log':
        ax[6].set_yscale('log')

    ax[7].boxplot(bl, patch_artist=True, medianprops=dict(color='k'), vert=True, meanline = True, showfliers= False, labels = label)
    ax[7].set_ylabel('BL [ms]')
    ax[7].set_title("Burstlength")
    ax[7].grid(which = "both", axis = 'y')
    ax[7].minorticks_on()
    if scale == 'log':
        ax[7].set_yscale('log')

    ax[8].boxplot(bl_median, patch_artist=True, medianprops=dict(color='k'), vert=True, meanline = True, showfliers= False, labels = label)
    ax[8].set_ylabel('BL [ms]')
    ax[8].set_title("median Burstlength")
    ax[8].grid(which = "both", axis = 'y')
    ax[8].minorticks_on()
    if scale == 'log':
        ax[8].set_yscale('log')

    ax[9].boxplot(bi, patch_artist=True, medianprops=dict(color='k'), vert=True, meanline = True, showfliers= False, labels = label)
    ax[9].set_ylabel('BI')
    ax[9].set_title("Burstiness Index")
    ax[9].grid(which = "both", axis = 'y')
    ax[9].minorticks_on()
    if scale == 'log':
        ax[9].set_yscale('log')

    for config in range(len(cv_mean)):
        ax[10].scatter(cv_mean[config], cv_std[config], s=1, label = label[config])
    maxval = max([max(i) for i in [*cv_mean, *cv_std]])
    minval = min([min(i) for i in [*cv_mean, *cv_std]])
    ax[10].plot([minval,maxval],[minval,maxval], color = 'black', linestyle='--', linewidth=1)
    ax[10].set_xlim([minval, maxval])
    ax[10].set_ylim([minval, maxval])
    ax[10].set_xscale('log')
    ax[10].set_yscale('log')
    ax[10].legend(fontsize=10)
    ax[10].set_xlabel('mean')
    ax[10].set_ylabel('std.dev.')

    plt.tight_layout()
    for t in ext:
        plt.savefig(os.path.join(path, fname + "." + t), format = t, bbox_inches = "tight")
    plt.close(fig)

def analyse_configs():

    return