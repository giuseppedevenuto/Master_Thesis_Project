% questo codice plotta due fasi successive alla volta
% NON valido se ci sono delle fasi intermedie mancanti, per esempio nel
% gruppo SHAM-SHAM e SHAM-CL

clear
clc
close all

wdwsize_time = 0.8; % sec - wdw after stimulus
binsize_time = 0.004; % sec
y_lim=200;

ele_array = [1, 7, 13, 14;
    3, 4, 10, 16;
    2, 8, 12, 11;
    6, 5, 9, 15];

PrePost_PSTH_directory=uigetdir('C:\Users\Giuseppe\Desktop','Select Directory Where To Save (la cartella coi PSTH del gruppo di ratti su cui lavorare)');
PrePost_PSTHgraph_directory=uigetdir(fullfile(PrePost_PSTH_directory,'\..\..'),'Select Directory Where To Save (la cartella in cui salvare i PSTH dei vari gruppi di ratti a due a due)');
%
rats_groups_file_directory=uigetdir('C:\Users\Giuseppe\Desktop','cartella col file in cui ci sono scritti i gruppi dei ratti');
rats_groups_file=uigetfile(fullfile(rats_groups_file_directory,'*.xlsx'));
table=importdata(fullfile(rats_groups_file_directory,rats_groups_file));
rats_groups_table=table(2:end,3:4);
clear table;

for i=1:size(rats_groups_table,1)
    if isempty(rats_groups_table{i,1})
        rats_groups_table{i,1}=rats_groups_table{i-1,1};
    end
end
%

rats = dir (fullfile(PrePost_PSTH_directory,'R*-*'));
ratnames={rats.name};
clear rats
nrats=length(ratnames);

for k=1:nrats
    maptypes= dir(fullfile(PrePost_PSTH_directory,ratnames{k},'cm*'));
    maptypenames={maptypes.name};
    clear maptypes
    nmaptypes=length(maptypenames);

    %
    for r=1:size(rats_groups_table,1)
        if ~any(rats_groups_table{r,2}~=ratnames{k})
            idx_rat = r;
        end
    end
    %

    save_dir=fullfile(PrePost_PSTHgraph_directory,rats_groups_table{idx_rat,1},ratnames{k});
    mkdir(fullfile(save_dir,'K1'))
    mkdir(fullfile(save_dir,'K2'))

    for h=1:nmaptypes
        for t=1:2
            switch t
                case 1
                    area1 = 'RFA';
                case 2
                    area1 = 'S1';
            end
            halfstims=dir(fullfile(PrePost_PSTH_directory,ratnames{k},maptypenames{h},area1,'PrePostPSTH*'));
            halfstimsnames={halfstims.name};
            clear halfstims
            nhalfstims=length(halfstimsnames);

            switch maptypenames{h}
                case "cm"
                    mapping="1Hz";
                case "cm2"
                    mapping="0.2Hz";
            end

            for g=1:nhalfstims

                electrodes = dir(fullfile(PrePost_PSTH_directory,ratnames{k},maptypenames{h},area1,halfstimsnames{g},'*.mat'));
                electrodenames = {electrodes.name};
                PSTH_electrodes_directory = electrodes.folder;
                clear electrodes
                nelectrodes = length(electrodenames);

                X=(0:wdwsize_time/binsize_time-1)*0.004;

                tempfig4x4=cell(nelectrodes,1);

                for n=1:nelectrodes
                    load(fullfile(PSTH_electrodes_directory,electrodenames{n}));

                    tempfigsingleele=cell(length(PrePostPSTH{1})-1,1);
                    for f=1:length(PrePostPSTH{1})-1
                        tempfigsingleele{f}=PrePostPSTH{2}(f:f+1,:);
                    end

                    tempfig4x4{n}=tempfigsingleele;
                end

                for fig=1:length(PrePostPSTH{1})-1

                    electrodesPSTHfigure=figure(Visible="off");
                    subp_elec = tiledlayout(ceil(sqrt(nelectrodes)),ceil(sqrt(nelectrodes)),"TileSpacing","tight","Padding","tight");

                    for numele=1:nelectrodes

                        time = 0:binsize_time:(wdwsize_time-binsize_time);

                        for m=1:size(tempfig4x4{numele}{fig},1)
                            electrodePSTHareas=trapz(X, tempfig4x4{numele}{fig}(m,:));
                            if electrodePSTHareas==0
                                tempfig4x4{numele}{fig}(m,tempfig4x4{numele}{fig}(m,:)==0) = NaN;
                            end
                        end

                        currentTile = nexttile(find(ele_array'==numele));

                        plot(time,tempfig4x4{numele}{fig}(1,:))
                        hold on
                        plot(time,tempfig4x4{numele}{fig}(2,:))
                        hold off

                        box off
                        xticks(0:(wdwsize_time/2):wdwsize_time)
                        xlim([0 wdwsize_time])
                        ylim([0 y_lim])
                        title(strcat('Channel',num2str(numele,'%02.f')))

                        if t == 2 && numele == (where_stim_ch-16)

                            box on

                            currentTile.LineWidth = 1;
                            currentTile.XColor = [1 0 0];
                            currentTile.YColor = [1 0 0];
                        end
                    end

                    currentTile = nexttile(1);
                    if fig==1
                        lgd = legend({'PreL','PosL'});
                        phases='PreLPosL';
                    else
                        lgd = legend({'PosL','PoS'});
                        phases='PosLPoS';
                    end
                    lgd.FontSize = 5;
                    legend box off
                    ylabel(subp_elec,'Spike Frequency [#sp/s]')
                    xlabel(subp_elec,'Time [s]')

                    Rats=cellfun(@(s)s(end-1:end),cellstr(ratnames{k}),'uni',0);

                    Stims=cellfun(@(s)s(end-4:end),cellstr(halfstimsnames{g}),'uni',0);
                    switch Stims{1}
                        case "01-30"
                            ch_stim="K1";
                        case "31-60"
                            ch_stim="K2";
                    end

                    title(subp_elec,strcat("Rat-",Rats{1}," ",ch_stim," ",mapping," ",area1))

                    exportgraphics(electrodesPSTHfigure,fullfile(save_dir,ch_stim,strcat('PSTH_',ratnames{k},"_",ch_stim,"_",mapping,"_",area1,"_",phases,".png")),'Resolution',500)
                    electrodesPSTHfigure.Visible="on";
                    saveas(electrodesPSTHfigure,fullfile(save_dir,ch_stim,strcat('PSTH_',ratnames{k},"_",ch_stim,"_",mapping,"_",area1,"_",phases,".fig")))
                    close(electrodesPSTHfigure)

                end
            end
        end
    end
end