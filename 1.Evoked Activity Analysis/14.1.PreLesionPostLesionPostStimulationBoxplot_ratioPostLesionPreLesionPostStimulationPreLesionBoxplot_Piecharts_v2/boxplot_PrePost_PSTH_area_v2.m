% plot box plot assoluti, boxplot relativi al prelesione e pie relativi al
% numero di campioni che supera la media del prelesione o è sotto
% --> in questa seconda versione ogni post lesione e post stim viene
% normalizzata per la media del proprio pre lesione
% --> in quello vecchio si considerava la normalizzazione su tutti i ratti
% messi insieme 

close all
clear
clc

nareas_x_nmapping=4;
ncolumn = 5;    % 1) colonna con le aree dei PSTH prima della stim; 2) colonna con le aree dei PSTH dopo la stim;
                % 3) colonna con area registrata e tipo di mapping; 4) nome delle fasi; 5) nome delle stim
nphasemax = 3;
nstimphase = 2;
nstimtypes = 3;

PrePost_PSTHarea_directory=uigetdir('C:\Users\Giuseppe\Desktop','Select Directory Where To Load (la cartella coi gruppi di ratti su cui lavorare)');
PrePost_PSTHareagraph_boxplot_directory=uigetdir(fullfile(PrePost_PSTHarea_directory,'\..'),'Select Directory Where To Save (la cartella in cui salvare gli scatterplot delle aree dei PSTH per tipo di lesione)');

for l=1:2
    switch l
        case 1
            lesion = 'ET1-OL';
        case 2
            lesion = 'SHAM-OL';
    end
    groups = dir(fullfile(PrePost_PSTHarea_directory,strcat(lesion,'*')));
    groupnames = {groups.name};
    clear groups
    ngroups = length(groupnames);

    datatoplot = cell(nareas_x_nmapping,ncolumn);

    for p=1:ngroups

        rats = dir (fullfile(PrePost_PSTHarea_directory,groupnames{p},'R*-*'));
        ratnames={rats.name};
        clear rats
        nrats=length(ratnames);

        for k=1:nrats

            maptypes= dir(fullfile(PrePost_PSTHarea_directory,groupnames{p},ratnames{k},'*Hz'));
            maptypenames={maptypes.name};
            clear maptypes
            nmaptypes=length(maptypenames);

            for h=1:nmaptypes
                for t=1:2
                    switch t
                        case 1
                            area1 = 'RFA';
                        case 2
                            area1 = 'S1';
                    end

                    stimchs = dir(fullfile(PrePost_PSTHarea_directory,groupnames{p},ratnames{k},maptypenames{h},area1,'*.mat'));
                    stimchnames = {stimchs.name};
                    filetoload_directory = stimchs.folder;
                    clear stimchs
                    nstimchs = length(stimchnames);

                    for s=1:nstimchs

                        load(fullfile(filetoload_directory,stimchnames{s}));

                        switch t
                            case 1
                                switch maptypenames{h}
                                    case "0.2Hz"
                                        row=3;
                                    case "1Hz"
                                        row=1;
                                end

                            case 2
                                switch maptypenames{h}
                                    case "0.2Hz"
                                        row=4;
                                    case "1Hz"
                                        row=2;
                                end
                        end

                        temp_PreS_phases=zeros(size(electrodesPSTHareas_NaN,1),nphasemax-length(nstimphase));
                        temp_PoS_phases=zeros(size(electrodesPSTHareas_NaN,1),nstimtypes);

                        type = cellfun(@(s)s((length(lesion)+2):end),cellstr(groupnames{p}),'uni',0);
                        switch type{1}
                            case 'EXP'
                                ntype=1;

                            case 'RS'
                                ntype=2;

                            case 'SH'
                                ntype=3;
                        end

                        for phase=1:length(mapphases)

                            if str2double(mapphases{phase}(2))<nstimphase
                                col_phase=str2double(mapphases{phase}(2))+1;
                                temp_PreS_phases(:,col_phase) = electrodesPSTHareas_NaN(:,phase);
                            else
                                temp_PoS_phases(:,ntype) = electrodesPSTHareas_NaN(:,phase);

                            end
                            
                        end

                        datatoplot{row,1}(end+1:end+size(electrodesPSTHareas_NaN,1),:) = temp_PreS_phases;
                        datatoplot{row,2}(end+1:end+size(electrodesPSTHareas_NaN,1),:) = temp_PoS_phases;

                        datatoplot{row,3} = strcat(area1," ",maptypenames{h});

                        if length(datatoplot{row,4})~=nphasemax && length(datatoplot{row,4})<length(mapphases)
                            datatoplot{row,4} = mapphases;
                        end

                        datatoplot{row,5}(1,ntype) = type;

                    end
                end
            end
        end
    end
%------

    groups_box_PSTHarea_figure=figure(Visible="off");
    subp_box = tiledlayout(ceil(sqrt(nareas_x_nmapping)),ceil(sqrt(nareas_x_nmapping)),"TileSpacing","tight","Padding","tight");

    for n=1:nareas_x_nmapping
        datatoplot{n,1}(datatoplot{n,1}==0) = NaN;
        datatoplot{n,2}(datatoplot{n,2}==0) = NaN;

        currentTile = nexttile(n);

        boxplot([datatoplot{n,1}, ...
            [datatoplot{n,1}(~isnan(datatoplot{n,2}(:,1)),:);NaN(length(datatoplot{n,1})-length(datatoplot{n,1}(~isnan(datatoplot{n,2}(:,1)),:)),2)],datatoplot{n,2}(:,1), ...
            [datatoplot{n,1}(~isnan(datatoplot{n,2}(:,2)),:);NaN(length(datatoplot{n,1})-length(datatoplot{n,1}(~isnan(datatoplot{n,2}(:,2)),:)),2)],datatoplot{n,2}(:,2), ...
            [datatoplot{n,1}(~isnan(datatoplot{n,2}(:,3)),:);NaN(length(datatoplot{n,1})-length(datatoplot{n,1}(~isnan(datatoplot{n,2}(:,3)),:)),2)],datatoplot{n,2}(:,3)],'Positions',[1,1.5, 2.5,3,3.5, 4.5,5,5.5, 6.5,7,7.5])
        xticklabels({})

        box off
        title(datatoplot{n,3})
        ylim([-5 60])

    end

    xticklabels(nexttile(3),['PreL','PosL',strcat(repmat({'PreL','PosL','PoS'},1,nstimtypes)," (",reshape(repmat(datatoplot{n,5}',1,nphasemax)',1,[]),')')])
    xticklabels(nexttile(4),['PreL','PosL',strcat(repmat({'PreL','PosL','PoS'},1,nstimtypes)," (",reshape(repmat(datatoplot{n,5}',1,nphasemax)',1,[]),')')])

    title(subp_box,strcat(lesion))
    xlabel(subp_box,'phases')
    ylabel(subp_box,'PSTH areas')
    exportgraphics(groups_box_PSTHarea_figure,fullfile(PrePost_PSTHareagraph_boxplot_directory,strcat('PSTHareas_',lesion,".png")),'Resolution',500)
    groups_box_PSTHarea_figure.Visible="on";
    saveas(groups_box_PSTHarea_figure,fullfile(PrePost_PSTHareagraph_boxplot_directory,strcat('PSTHareas_',lesion,".fig")))
    close(groups_box_PSTHarea_figure)
%-------------------------------
    groups_ratiobox_PSTHarea_figure=figure(Visible="off");
    subp_ratiobox = tiledlayout(ceil(sqrt(nareas_x_nmapping)),ceil(sqrt(nareas_x_nmapping)),"TileSpacing","tight","Padding","tight");

    for n=1:nareas_x_nmapping
        meanPreL_total=mean(datatoplot{n,1}(~isnan(datatoplot{n,1}(:,1)),1));
        meanPreL_exp=mean(datatoplot{n,1}(and(~isnan(datatoplot{n,2}(:,1)),~isnan(datatoplot{n,1}(:,1))),1));
        meanPreL_rs=mean(datatoplot{n,1}(and(~isnan(datatoplot{n,2}(:,2)),~isnan(datatoplot{n,1}(:,1))),1));
        meanPreL_sh=mean(datatoplot{n,1}(and(~isnan(datatoplot{n,2}(:,3)),~isnan(datatoplot{n,1}(:,1))),1));

        currentTile = nexttile(n);

        %------------
        boxplot([datatoplot{n,1}(:,2)/meanPreL_total, ...
            [datatoplot{n,1}(~isnan(datatoplot{n,2}(:,1)),2);NaN(length(datatoplot{n,1})-length(datatoplot{n,1}(~isnan(datatoplot{n,2}(:,1)),2)),1)]/meanPreL_exp,datatoplot{n,2}(:,1)/meanPreL_exp, ...
            [datatoplot{n,1}(~isnan(datatoplot{n,2}(:,2)),2);NaN(length(datatoplot{n,1})-length(datatoplot{n,1}(~isnan(datatoplot{n,2}(:,2)),2)),1)]/meanPreL_rs,datatoplot{n,2}(:,2)/meanPreL_rs, ...
            [datatoplot{n,1}(~isnan(datatoplot{n,2}(:,3)),2);NaN(length(datatoplot{n,1})-length(datatoplot{n,1}(~isnan(datatoplot{n,2}(:,3)),2)),1)]/meanPreL_sh,datatoplot{n,2}(:,3)/meanPreL_sh],'Positions',[1, 2,2.5, 3.5,4, 5,5.5])
        %------------

        % boxplot([datatoplot{n,1}(:,2)/meanPreL_total,datatoplot{n,2}/meanPreL_total],['PosL',strcat("PoS (",datatoplot{n,5},')')])
        xticklabels({})
        hold on
        yline(1,'--')

        box off
        title(datatoplot{n,3})
        ylim([-0.1 3.2])

    end

    %------------
    xticklabels(nexttile(3),['PosL',strcat(repmat({'PosL','PoS'},1,nstimtypes)," (",reshape(repmat(datatoplot{n,5}',1,nphasemax-1)',1,[]),')')])
    xticklabels(nexttile(4),['PosL',strcat(repmat({'PosL','PoS'},1,nstimtypes)," (",reshape(repmat(datatoplot{n,5}',1,nphasemax-1)',1,[]),')')])
    %------------

    % xticklabels(nexttile(3),['PosL',strcat("PoS (",datatoplot{n,5},')')])
    % xticklabels(nexttile(4),['PosL',strcat("PoS (",datatoplot{n,5},')')])

    title(subp_ratiobox,strcat(lesion))
    xlabel(subp_ratiobox,'phases')
    ylabel(subp_ratiobox,'ratio PSTH areas')
    exportgraphics(groups_ratiobox_PSTHarea_figure,fullfile(PrePost_PSTHareagraph_boxplot_directory,strcat('ratioPSTHareas_',lesion,".png")),'Resolution',500)
    groups_ratiobox_PSTHarea_figure.Visible="on";
    saveas(groups_ratiobox_PSTHarea_figure,fullfile(PrePost_PSTHareagraph_boxplot_directory,strcat('ratioPSTHareas_',lesion,".fig")))
    close(groups_ratiobox_PSTHarea_figure)
%--------------

    for n=1:nareas_x_nmapping

    groups_piechart_PSTHarea_figure=figure(Visible="off");
    subp_pie = tiledlayout(2,4,"TileSpacing","tight","Padding","compact");

    meanPreL_total=mean(datatoplot{n,1}(~isnan(datatoplot{n,1}(:,1)),1));
    meanPreL_exp=mean(datatoplot{n,1}(and(~isnan(datatoplot{n,2}(:,1)),~isnan(datatoplot{n,1}(:,1))),1));
    meanPreL_rs=mean(datatoplot{n,1}(and(~isnan(datatoplot{n,2}(:,2)),~isnan(datatoplot{n,1}(:,1))),1));
    meanPreL_sh=mean(datatoplot{n,1}(and(~isnan(datatoplot{n,2}(:,3)),~isnan(datatoplot{n,1}(:,1))),1));

    all_mean=[meanPreL_total,meanPreL_exp,meanPreL_rs,meanPreL_sh];
    all_phases=[datatoplot{n,1}(:,2), ...
            [datatoplot{n,1}(~isnan(datatoplot{n,2}(:,1)),2);NaN(length(datatoplot{n,1})-length(datatoplot{n,1}(~isnan(datatoplot{n,2}(:,1)),2)),1)],datatoplot{n,2}(:,1), ...
            [datatoplot{n,1}(~isnan(datatoplot{n,2}(:,2)),2);NaN(length(datatoplot{n,1})-length(datatoplot{n,1}(~isnan(datatoplot{n,2}(:,2)),2)),1)],datatoplot{n,2}(:,2), ...
            [datatoplot{n,1}(~isnan(datatoplot{n,2}(:,3)),2);NaN(length(datatoplot{n,1})-length(datatoplot{n,1}(~isnan(datatoplot{n,2}(:,3)),2)),1)],datatoplot{n,2}(:,3)];
    all_phases_names=['PosL',strcat(repmat({'PosL','PoS'},1,nstimtypes)," (",reshape(repmat(datatoplot{n,5}',1,nphasemax-1)',1,[]),')')];

    for npie=3:8

        currentTile = nexttile(npie);
        
        greater=sum(all_phases(~isnan(all_phases(:,npie-1)),npie-1)>all_mean(ceil(npie/2)))/sum(~isnan(all_phases(:,npie-1)));

        pie([greater,1-greater]);

        title(all_phases_names(npie-1))

    end 

    nexttile(1:2);
    greater=sum(all_phases(~isnan(all_phases(:,1)),1)>all_mean(1))/sum(~isnan(all_phases(:,1)));
    pie([greater,1-greater]);
    title(all_phases_names(1))
    legend({'> 1','< 1'})
    legend box off

    title(subp_pie,strcat(lesion," ",datatoplot{n,3}))

    exportgraphics(groups_piechart_PSTHarea_figure,fullfile(PrePost_PSTHareagraph_boxplot_directory,strcat('piePSTHareas_',lesion,'_',datatoplot{n,3},".png")),'Resolution',500)
    groups_piechart_PSTHarea_figure.Visible="on";
    saveas(groups_piechart_PSTHarea_figure,fullfile(PrePost_PSTHareagraph_boxplot_directory,strcat('piePSTHareas_',lesion,'_',datatoplot{n,3},".fig")))
    close(groups_piechart_PSTHarea_figure)

    end

end