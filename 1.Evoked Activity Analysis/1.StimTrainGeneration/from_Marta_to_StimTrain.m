% codice che estrae dal treno di stimoli di onde quadre i diversi tipi di
% stim train (fronte di salita, fronte di discesa o fronte di inversione
% del segno)

n_ch_stimulated = 2; % numero di psth per registrazione che dipende dal numero di canali di stimolazione
load_directory = uigetdir('C:\Users\Giuseppe\Desktop','Select Directory Where To Work (la cartella del gruppo di ratti su cui lavorare)');
save_directory_start = uigetdir('C:\Users\Giuseppe\Desktop','Select Directory Where To Save (la cartella col nome del gruppo di ratti in cui salvare gli stim train start)');
%save_directory_rise = uigetdir('C:\Users\Giuseppe\Desktop','Select Directory Where To Save (la cartella col nome del gruppo di ratti in cui salvare gli stim train rise)');

% rilevazione del nome e del numero di ratti
rats = dir(fullfile(load_directory,'R*-*')); 
ratnames = {rats.name};
clear rats
nrats = length(ratnames);
for k = 1:nrats

    % rilevazione del nome e del numero di registrazioni fatte (fasi x tipi di mapping)
    mapmeasurements = dir(fullfile(load_directory,ratnames{k},'*0*cm*'));
    mapmeasurementnames = {mapmeasurements.name};
    clear mapmeasurements
    nmapmeasurements = length(mapmeasurementnames);
    for h = 1:nmapmeasurements

        % creazione cartelle in cui salvare i treni di stim
        mkdir(fullfile(save_directory_start,ratnames{k},strrep(mapmeasurementnames{h},'_','-')))
        % mkdir(fullfile(save_directory_rise,ratnames{k},strrep(mapmeasurementnames{h},'_','-')))

        stim_file = dir(fullfile(load_directory,ratnames{k},mapmeasurementnames{h},'*.mat'));
        stim_file_name = stim_file.name;
        stim_file_directory = stim_file.folder;
        load(fullfile(stim_file_directory,stim_file_name))

        % creazione e salvataggio del treno di stim considerando l'istante di inizio
        % dell'onda quadra
        [stim_train,where_stim,how_many_stim] = stim_detection_start(data,Channels,n_ch_stimulated);
        save(fullfile(save_directory_start,ratnames{k},strrep(mapmeasurementnames{h},'_','-'),'stim_data'),'stim_train','where_stim','how_many_stim')

        % creazione e salvataggio del treno di stim considerando l'istante
        % di passaggio da valore negativo a positivo dell'onda quadra
        % [stim_train,where_stim,how_many_stim] = stim_detection_rise(data,Channels,n_ch_stimulated);
        % save(fullfile(save_directory_rise,ratnames{k},strrep(mapmeasurementnames{h},'_','-'),'stim_data'),'stim_train','where_stim','how_many_stim')

    end
end