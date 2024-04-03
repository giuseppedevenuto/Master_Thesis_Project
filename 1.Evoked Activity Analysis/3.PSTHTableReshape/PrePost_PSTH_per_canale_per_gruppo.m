% prende in considerazione le matrici dei psth dei diversi canali (in cui 
% abbiamo in colonna i psth in base ai canali di stimolazione) di una sola 
% fase di un tipo di mapping e li mette insieme per costruire una matrice
% che sulle diverse righe abbia i psth di uno stesso canale e con la
% stimolazione dello stesso canale. In questo modo è possibile confrontare
% il psth delle diverse fasi pre-lesion, post-lesione e post-stimolazione.
% Ogni canale avrà tante matrici quanti sono i canali attraveso cui si è
% stimolato.

clear
clc

nelectrodes=16;
wdwsize_time = 0.8; % sec - wdw after stimulus
binsize_time = 0.004; % sec
nbin = wdwsize_time/binsize_time;       % total number of bin for the histogram

% -> directory to use
PSTH_directory=uigetdir('C:\Users\Giuseppe\Desktop','Select Directory Where To Load (la cartella coi PSTH del gruppo di ratti su cui lavorare)');
PrePost_PSTH_directory=uigetdir('C:\Users\Giuseppe\Desktop','Select Directory Where To Save (la cartella col nome del gruppo di ratti in cui salvare i PSTH Pre-Post)');

%--> read rats names and number of rats
rats = dir (fullfile(PSTH_directory,'R*-*'));
ratnames={rats.name};
clear rats
nrats=length(ratnames);
for k=1:nrats
    %--> read types of mapping available and number of mapping
    maptypes= dir(fullfile(PSTH_directory,ratnames{k},'00-cm*'));
    maptypenames={maptypes.name};
    clear maptypes
    nmaptypes=length(maptypenames);
    for h=1:nmaptypes
        % -> read name of the mapping type
        maptypename=cellfun(@(s)s(4:6),cellstr(maptypenames{h}),'uni',0);
        % -> read name of the phases for the mapping type and number of
        % phases
        mapphases=dir(fullfile(PSTH_directory,ratnames{k},strcat('*',maptypename{1},'*')));
        mapphasenames={mapphases.name};
        clear mapphases
        nmapphases=length(mapphasenames);
        for t=1:2 %--> recorded areas
            switch t
                case 1
                    area1 = 'RFA';
                case 2
                    area1 = 'S1';
            end

            %--> make directory to save data
            mkdir(fullfile(PrePost_PSTH_directory,ratnames{k},strrep(maptypename{1},'-',''),area1,'PrePostPSTH 01-30'))
            mkdir(fullfile(PrePost_PSTH_directory,ratnames{k},strrep(maptypename{1},'-',''),area1,'PrePostPSTH 31-60'))

            for n=1:nelectrodes
                mapphasenametosave=cell(nmapphases,1); % salva il nome della phase a cui appartengono i psth
                % -> due matrici, tante quanti sono i canali su cui si è stimolato
                PSTH_1_30=zeros(nmapphases,nbin); 
                PSTH_31_60=zeros(nmapphases,nbin);

                for m=1:nmapphases
                    % cicla sulle fasi e mette in colonna i vari psth di
                    % uno stesso tipo di mapping
                    electrodes = dir(fullfile(PSTH_directory,ratnames{k},mapphasenames{m},area1,'*.mat'));
                    electrodenames = {electrodes.name};
                    PSTH_electrodes_directory = electrodes.folder;
                    clear electrodes
                    load(fullfile(PSTH_electrodes_directory,electrodenames{n})) %-> carica la matrice da cui prendere la riga(canale) di interesse

                    mapphasename = cellfun(@(s)s(1:2),cellstr(mapphasenames{m}),'uni',0);
                    mapphasenametosave{m} = mapphasename{1}(:); %-> assegna il nome
                    %-> assegna i psth alle righe corrispondenti
                    PSTH_1_30(m,:)=psth_FR_vectors(1,:); 
                    PSTH_31_60(m,:)=psth_FR_vectors(2,:);
                end

                % -> salva i diversi file in due cartelle diverse
                PrePostPSTH= {mapphasenametosave,PSTH_1_30};
                where_stim_ch = where_stim(1);
                save(fullfile(PrePost_PSTH_directory,ratnames{k},strrep(maptypename{1},'-',''),area1,'PrePostPSTH 01-30',strcat('PrePostPSTH_01-30_Ch_',num2str(n,'%02.f'))),'PrePostPSTH','where_stim_ch')

                PrePostPSTH= {mapphasenametosave,PSTH_31_60};
                where_stim_ch = where_stim(2);
                save(fullfile(PrePost_PSTH_directory,ratnames{k},strrep(maptypename{1},'-',''),area1,'PrePostPSTH 31-60',strcat('PrePostPSTH_31-60_Ch_',num2str(n,'%02.f'))),'PrePostPSTH','where_stim_ch')

            end
        end
    end
end

