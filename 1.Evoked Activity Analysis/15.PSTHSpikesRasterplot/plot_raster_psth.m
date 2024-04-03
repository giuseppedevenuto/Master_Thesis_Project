clear
clc

fs = 25000; % sampling rate [kHz]

wdwsize_time = 0.8; % sec - wdw after stimulus
binsize_time = 0.004; % sec
artremoval_wdwsize_time = 0.0004; % sec

ele = 8; % choose n. electrode
k = 2; % choose electrode of stimulation
experimental_group = 'ET1-OL'; % choose experimental group
rat = 'R21-32'; % choose rat

%--> directory to use
spike_train_directory = uigetdir('C:\Users\Giuseppe\Desktop','Select Directory Where To Work (la cartella con gli spike train del gruppo di ratti su cui lavorare)');
stim_train_directory = uigetdir('C:\Users\Giuseppe\Desktop','Select Directory Where To Work (la cartella con gli stim train del gruppo di ratti su cui lavorare)');
save_PSTH_directory = uigetdir('C:\Users\Giuseppe\Desktop','Select Directory Where To Save (la cartella col nome del gruppo di ratti in cui salvare i PSTH)');
%%
phases = dir(fullfile(spike_train_directory,experimental_group,rat,'0*-cm2*'));
phasenames = {phases.name};
clear phases
nphases = length(phasenames);

dim = [100 100 700 500];
raster_psth = figure;
set(gcf,'position',dim)
raster_imm = tiledlayout(3,2,"TileSpacing","compact","Padding","tight");

colour = {  [0      0.4470 0.7410],...
            [0.8500 0.3250 0.0980],...
            [0.9290 0.6940 0.1250]};
for p=1:nphases
    dir_stim = dir(fullfile(stim_train_directory,experimental_group,rat,strcat('*',phasenames{p}(1:20)),'*.mat'));
    load(fullfile(dir_stim.folder,dir_stim.name))
    for t=1:2 %--> recorded areas
        switch t
            case 1
                area1 = 'RFA';
            case 2
                area1 = 'S1';
        end
        %--> read name of clusters for each electrode
        clusters = dir(fullfile(spike_train_directory,experimental_group,rat,phasenames{p},strcat(phasenames{p},'*'),area1,strcat('*_',num2str(ele,'%02.f'),'_*.mat')));
        clusternames = {clusters.name};
        clusters_directory = clusters.folder;
        clear clusters
        nclusters = length(clusternames);
        spike_train_clusters = [];
        for g = 1:nclusters %--> load all the cluster in one matrix
            load(fullfile(clusters_directory,clusternames{g}))
            clear artifact spikes SPK_CLU
            spike_train_clusters = [spike_train_clusters, peak_train];
            clear peak_train
        end
        %--> merge all the cluster in one spike train
        spike_train = double(sum(spike_train_clusters,2)>=1);
        clear spike_train_clusters

        %--> calcola il PSTH
        xyraster = rasterplot_psth(spike_train,stim_train,fs,binsize_time,artremoval_wdwsize_time,wdwsize_time,length(where_stim),how_many_stim,k);
        currentTile = nexttile(t+(p-1)*2, [1 1]);
        scatter(xyraster(:,1),xyraster(:,2),[],colour{p},'Marker','.') 
        % set(currentTile, 'XTick', [], 'XTickLabel', []);
        set(currentTile, 'YTick', [], 'YTickLabel', []);
        % set(get(currentTile, 'XAxis'), 'Visible', 'off');
        % set(get(currentTile, 'YAxis'), 'Visible', 'off');
        ylabel(currentTile, 'Trial')
        xlabel(currentTile, 'Time [ s ]')
    end
end

exportgraphics(raster_psth,fullfile(save_PSTH_directory,strcat("RasterPSTH_2.png")),'Resolution',500)
saveas(raster_psth,fullfile(save_PSTH_directory,strcat("RasterPSTH_2.fig")))
close(raster_psth)