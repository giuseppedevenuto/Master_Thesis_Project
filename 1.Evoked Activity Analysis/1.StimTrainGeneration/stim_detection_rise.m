function [single_stim_train,sorted_true_active_ch,sorted_n_stim] = stim_detection_rise(pulse_train,idx_active_channels,n_stim_ch)
% crea il treno di stimolazioni quando lo stimolo passa da valori negativi a
% positivi

%%% Output
% single_stim_train         = stim train
% sorted_true_active_ch     = canali effettivi attraveso cui è stata fatta la
%                             stimolazione in ordine temporale 
% sorted_n_stim             = numero di stimolazioni date su ogni canale effettivo di
%                             stimolazione in ordine temporale

%%% Input
% idx_active_channels       = canali attraverso cui è stata fatta la stimolazione;
% pulse_train               = treno di onde quadre;
% n_stim_ch                 = numero di canali attraverso cui è stata fatta la stimolazione;

active_ch = find(idx_active_channels); % trova indice dei canali di stimolazione (da 1 a 32)
clear idx_active_channels
stim_train = zeros(length(active_ch),length(pulse_train)); % inizializza tanti stim train (con zeri) quanti sono i canali di stimolazione
first_stim_idx = zeros(length(active_ch),1); % inizializza il vettore che contiene le posizioni del primo stimolo di ogni canale di stim

for ch=1:length(active_ch)

    % calcolo del treno di stim
    idx_stim = diff(pulse_train(active_ch(ch),:))>0;
    stim_train(ch,:) = [0, full(idx_stim)]; % converte in double
    first_stim_idx(ch,:) = find(idx_stim,1); % trava la posizione della prima stim
    clear idx_stim

end
clear pulse_train ch
[n_stim,true_ch] = sort(sum(stim_train,2),'descend'); % ordina i treni di stim in modo discendente considerando il numero di stim (in alcuni c'è solo una stim)
single_stim_train = sum(stim_train(true_ch(1:n_stim_ch),:)); % crea un treno di stim singolo considerando solo i canali con più stimolazioni
clear stim_train
true_active_ch = active_ch(true_ch(1:n_stim_ch),:); % prende solo il numero dei canali reali di stim (quelli attraverso cui si son dati più stimoli)
clear active_ch
true_first_stim_idx = first_stim_idx(true_ch(1:n_stim_ch),:); % prende solo le posizioni della prima stim dei canali reali di stim
clear first_stim_idx true_ch n_stim_ch
[~,ch_temporal_order] = sort(true_first_stim_idx); % ordina le posizioni della prima stim per canale per trovare quale sia il primo canale a stimolare 
clear true_first_stim_idx
sorted_true_active_ch = true_active_ch(ch_temporal_order); % trova l'ordine di stimolazione dei canali (mette in ordine il numero del canale in base all'ordine temporale)
clear true_active_ch
sorted_n_stim = n_stim(ch_temporal_order); % prende solo il numero di stimolazioni fatte su ogni canale dei canali reali di stimolazione messi in ordine
clear n_stim ch_temporal_order