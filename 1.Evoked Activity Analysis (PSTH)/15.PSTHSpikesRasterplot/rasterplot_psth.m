function xyraster=rasterplot_psth(peak_train, stim_train, fs, binsize_time, cancw, wdwsize_time, npsth, stimperpsth, k)
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
xyraster = [];
for e = (1:stimperpsth(k))+sum(stimperpsth(1:k-1))
    % Post Stimulus Time Histogram construction
    StimWin = peak_train(artifact(e):artifact(e)+wdwsize_samples-1);   % post-stimulus window
    StimWin(1:cancsample) = zeros(cancsample,1);                        % artifact blanking
    peak_index = find(StimWin)/fs;
    y = (e-sum(stimperpsth(1:k-1)))*ones(size(peak_index));
    xyraster = [xyraster;[reshape(peak_index,[],1) reshape(y,[],1)]];
end

end