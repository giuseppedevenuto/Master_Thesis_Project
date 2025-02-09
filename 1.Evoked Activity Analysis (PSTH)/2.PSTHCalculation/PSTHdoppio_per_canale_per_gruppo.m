% prendendo in considerazione i treni di spike e i treni di stimoli vengono
% creati i PSTH che consistono in una matrice dove il numero di righe
% definisce il numero di canali attraverso cui si è stimolato e il numero
% di colonne definisce il numero di bin del PSTH. Quindi i PSTH vengono
% costruiti prendendo in considerazione i diversi canali di stimolazione.

clear
clc

fs = 25000; % sampling rate [kHz]

wdwsize_time = 0.8; % sec - wdw after stimulus
binsize_time = 0.004; % sec
artremoval_wdwsize_time = 0.0004; % sec

nelectrodes = 16; % n. electrodes

%--> directory to use
spike_train_directory = uigetdir('C:\Users\Giuseppe\Desktop','Select Directory Where To Work (la cartella con gli spike train del gruppo di ratti su cui lavorare)');
stim_train_directory = uigetdir('C:\Users\Giuseppe\Desktop','Select Directory Where To Work (la cartella con gli stim train del gruppo di ratti su cui lavorare)');
save_PSTH_directory = uigetdir('C:\Users\Giuseppe\Desktop','Select Directory Where To Save (la cartella col nome del gruppo di ratti in cui salvare i PSTH)');

%--> read rats names 
rats = dir(fullfile(spike_train_directory,'R*-*'));
ratnames = {rats.name};
clear rats
nrats = length(ratnames);
for k = 1:nrats
    %--> read phase of mapping available
    mapmeasurements = dir(fullfile(spike_train_directory,ratnames{k},'0*-cm*'));
    mapmeasurementnames = {mapmeasurements.name};
    clear mapmeasurements
    nmapmeasurements = length(mapmeasurementnames);
    for h = 1:nmapmeasurements

        %--> make directory to save data
        mkdir(fullfile(save_PSTH_directory,ratnames{k},mapmeasurementnames{h},'\RFA'))
        mkdir(fullfile(save_PSTH_directory,ratnames{k},mapmeasurementnames{h},'\S1'))

        %--> load stim train
        dir_stim = dir(fullfile(stim_train_directory,ratnames{k},strcat('*',mapmeasurementnames{h}),'*.mat'));
        load(fullfile(dir_stim.folder,dir_stim.name))

        for t=1:2 %--> recorded areas
            switch t
                case 1
                    area1 = 'RFA';
                case 2
                    area1 = 'S1';
            end

            for n = 1:nelectrodes

                %--> read name of clusters for each electrode
                clusters = dir(fullfile(spike_train_directory,ratnames{k},mapmeasurementnames{h},strcat(mapmeasurementnames{h},'*'),area1,strcat('*_',num2str(n,'%02.f'),'_*.mat')));
                clusternames = {clusters.name};

                if ~isempty(clusternames) %--> check if there is at least a cluster for each electrode

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
                    spike_train = ceil(sum(spike_train_clusters,2));
                    clear spike_train_clusters

                    %--> calcola il PSTH
                    psth_output = PSTH_v2(spike_train,stim_train,fs,binsize_time,artremoval_wdwsize_time,wdwsize_time,length(where_stim),how_many_stim);
                    clear spike_train
                    psth_FR_vectors = psth_output/binsize_time;
                    clear psth_output

                    %--> salva il PSTH
                    save(fullfile(save_PSTH_directory,ratnames{k},mapmeasurementnames{h},area1,strcat(mapmeasurementnames{h},'_doublePSTH_1-30_31-60_Ch_',num2str(n,'%02.f'))),'psth_FR_vectors','where_stim')
                    clear psth_FR_vectors

                else %--> if not save an empty fake file

                    %--> calcola il PSTH fittizio
                    psth_FR_vectors = zeros(length(where_stim),wdwsize_time/binsize_time);
                    
                    %--> salva il PSTH fittizio
                    save(fullfile(save_PSTH_directory,ratnames{k},mapmeasurementnames{h},area1,strcat(mapmeasurementnames{h},'_doublePSTH_1-30_31-60_Ch_',num2str(n,'%02.f'))),'psth_FR_vectors','where_stim')
                    clear psth_FR_vectors

                end

            end
        end
    end
end