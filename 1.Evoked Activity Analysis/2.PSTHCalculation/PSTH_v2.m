function psth_output=PSTH_v2(peak_train, stim_train, fs, binsize_time, cancw, wdwsize_time, npsth, stimperpsth)
% calcola la matrice del PSTH prendendo in considerazione il numero di
% canali attraverso cui si è stimolato

%%% Output
% psth_count    = post-stimulus histogram matrix [number of spikes] (npsth x nbin)

%%% Input
% peak_train    = train of spikes with spike amplitude
% stim_train    = train with stimulus position
% fs            = sampling frequency [samples/sec]
% binsize_time  = PSTH bin - def by user [sec]
% cancw         = deleting artifact window - def by user [sec]
% wdwsize_time  = time length of the histogram [sec]
% npsth         = number of PSTH to built - def by the number of stimulation channels 
% stimperpsth   = number of stimuli per stimulation channel

% % % % % cancw = artremoval_wdwsize_time;
% % % % % stimperpsth = how_many_stim;

artifact = find(stim_train);
binsize_samples = binsize_time*fs;      % bin size [samples]
wdwsize_samples = wdwsize_time*fs;      % time length of the histogram [samples]
cancsample = cancw*fs;                  % canc window size [samples]
n = length(artifact);                   % number of artifact
nbin = wdwsize_time/binsize_time;       % total number of bin for the histogram
psth_output = zeros(npsth,nbin);        % matrix

% ----------> START PROCESSING
if (n >= 1)                         % number of artifact must be at least 1

    for t = 1:npsth
        psth_count = zeros(nbin,1);

        for k = (1:stimperpsth(t))+sum(stimperpsth(1:t-1))                      % cycle over stimuli
            % Post Stimulus Time Histogram construction
            StimWin = peak_train (artifact(k):artifact(k)+wdwsize_samples-1);   % post-stimulus window
            StimWin(1:cancsample) = zeros(cancsample,1);                        % artifact blanking
            peak_index = find(StimWin);                                         % index of spikes within stim_win
            bin_num = (ceil(peak_index/binsize_samples));                       % bin with spikes

            for i = 1:length(bin_num)
                psth_count (bin_num(i)) = 1 + psth_count (bin_num(i));          % fill in the bins in the histogram
            end
        end
        psth_output(t,:) = (psth_count/stimperpsth(t));                         % Normalization - Maximum is '1' only if the PSTHbin==PDbin
    end
end
end