% by G. De Venuto (06/11/2023) for in-vivo data
close all
clear
clc

spike_train_directory = uigetdir('C:\Users\Giuseppe\Desktop','Select Spike Train Directory with Rat Groups');
raster_directory = uigetdir('C:\Users\Giuseppe\Desktop','Select the folder where to save the ''raster'' file');

%--> read rats names
groups = dir(fullfile(spike_train_directory,'*-*'));
groupnames = {groups.name};
clear groups
ngroups = length(groupnames);
for g = 1:ngroups
    %--> read rats names
    rats = dir(fullfile(spike_train_directory,groupnames{g},'R*-*'));
    ratnames = {rats.name};
    clear rats
    nrats = length(ratnames);
    for r = 1:nrats
        %--> read phases names
        pphases = dir(fullfile(spike_train_directory,groupnames{g},ratnames{r},'00-basal1-*'));
        pphasenames = {pphases.name};
        clear pphases
        npphases = length(pphasenames);
        rec_time_stamp = zeros(1,npphases);
        % for pp = 1:npphases
        %     idx_ = strfind(pphasenames{pp},"_");
        %     rec_time_stamp(pp)=str2double(pphasenames{pp}((idx_(1)-6):(idx_(1)-1)));
        % end
        for pp = 1:npphases
            rec_time_stamp(pp)=str2double(pphasenames{pp}((end-5):(end)));
        end
        phase = dir(fullfile(spike_train_directory,groupnames{g},ratnames{r},strcat('00-basal1-*',num2str(min(rec_time_stamp)),'*'),strcat('00-basal1-*',num2str(min(rec_time_stamp)),'*')));
        phasedir = {phase.folder};
        phasename = {phase.name};
        clear phase
        %--> recorded areas
        for a=1:2
            switch a
                case 1
                    area = 'RFA';
                case 2
                    area = 'S1';
            end
            bnn_raster = [];
            %--> read neurons names
            neurons = dir(fullfile(phasedir{1},phasename{1},area,'*.mat'));
            neuronnames = {neurons.name};
            neurons_directory = neurons.folder;
            clear neurons
            nneurons = length(neuronnames);
            for n=1:nneurons
                %--> load peak train
                load(fullfile(neurons_directory,neuronnames{n}))
                time = find(peak_train); % in samples
                neuron_id = ones(size(time))*(n-1);
                bnn_raster = [bnn_raster; time, neuron_id];
            end

            writematrix(bnn_raster,fullfile(raster_directory,strcat(area,'_',ratnames{r},'_',phasename{1},'_bnn_raster.csv')))

        end
    end
end