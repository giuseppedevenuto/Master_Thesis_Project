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

mat_data = cell(2,4);
name_mat_data=cell(2,4);
ratio_mat = cell(2,4);
name_ratio_mat = cell(2,4);
pie_mat = cell(2,4);
name_pie_mat = cell(2,4);
statistical_test_matrix = cell(2,2);

for l=1:2

    cell_stat_test_matrix = cell(nareas_x_nmapping,ncolumn);
    switch l
        case 1
            lesion = 'ET1-OL';
        case 2
            lesion = 'SHAM-OL';
    end
    statistical_test_matrix{l,2}=lesion;
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

                    to_average = cell(2,1);
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

                        %-----stat test matrix
                        to_average{1}(end+1:end+size(electrodesPSTHareas_NaN,1),:) = temp_PreS_phases;
                        to_average{2}(end+1:end+size(electrodesPSTHareas_NaN,1),:) = temp_PoS_phases;

                    end

                    %-----stat test matrix
                    cell_stat_test_matrix{row,1}(end+1,:)=mean(FromZeroToNaN(to_average{1}),1,'omitnan');
                    cell_stat_test_matrix{row,2}(end+1,:)=mean(FromZeroToNaN(to_average{2}),1,'omitnan');

                    cell_stat_test_matrix{row,3} = strcat(area1," ",maptypenames{h});

                    if length(cell_stat_test_matrix{row,4})~=nphasemax && length(cell_stat_test_matrix{row,4})<length(mapphases)
                        cell_stat_test_matrix{row,4} = mapphases;
                    end

                    cell_stat_test_matrix{row,5}(1,ntype) = type;
                    %-----stat test matrix

                end
            end
        end
    end

    statistical_test_matrix{l,1}=cell_stat_test_matrix;
%------

    groups_box_PSTHarea_figure=figure(Visible="off");
    subp_box = tiledlayout(ceil(sqrt(nareas_x_nmapping)),ceil(sqrt(nareas_x_nmapping)),"TileSpacing","tight","Padding","tight");

    for n=1:nareas_x_nmapping
        datatoplot{n,1}(datatoplot{n,1}==0) = NaN;
        datatoplot{n,2}(datatoplot{n,2}==0) = NaN;

        currentTile = nexttile(n);
        
        ll_data = size(datatoplot{n,1},1); 
        x_pos = [1,1.5, 2.5,3,3.5, 4.5,5,5.5, 6.5,7,7.5];
        x_box_pos = x_pos+((1:numel(x_pos))-median(1:numel(x_pos)))*1/numel(x_pos);
        mat_to_plot = [datatoplot{n,1}, ...
            [datatoplot{n,1}(~isnan(datatoplot{n,2}(:,1)),:);NaN(length(datatoplot{n,1})-length(datatoplot{n,1}(~isnan(datatoplot{n,2}(:,1)),:)),2)],datatoplot{n,2}(:,1), ...
            [datatoplot{n,1}(~isnan(datatoplot{n,2}(:,2)),:);NaN(length(datatoplot{n,1})-length(datatoplot{n,1}(~isnan(datatoplot{n,2}(:,2)),:)),2)],datatoplot{n,2}(:,2), ...
            [datatoplot{n,1}(~isnan(datatoplot{n,2}(:,3)),:);NaN(length(datatoplot{n,1})-length(datatoplot{n,1}(~isnan(datatoplot{n,2}(:,3)),:)),2)],datatoplot{n,2}(:,3)];
        meanMatTot_init = mean(mat_to_plot, 1, 'omitnan');
        x_data_to_plot=ones(size(mat_to_plot));
        for colum=1:size(x_data_to_plot,2)
            x_data_to_plot(:,colum)=x_data_to_plot(:,colum)*x_pos(colum);
        end
        boxobj_init = boxchart(reshape(x_data_to_plot,1,[]), reshape(mat_to_plot,1,[]),'BoxWidth',3,'MarkerStyle','none','GroupByColor',reshape(x_data_to_plot,1,[]),'BoxFaceAlpha',1,'BoxMedianLineColor','k');
        box_col = [ 77 137 124; 
                    182 160 78;
                    77 137 124;
                    182 160 78;
                    154 195 232; 
                    77 137 124;
                    182 160 78;
                    154 195 232; 
                    77 137 124;
                    182 160 78;
                    154 195 232]/255;
        colororder(box_col)
        xticks(x_box_pos)

        xticklabels({})

        hold on
        for i=1:numel(x_box_pos)
            plot(x_box_pos(i), meanMatTot_init(i),'Marker','o','MarkerSize',3, 'LineStyle','none','MarkerEdgeColor','k','LineWidth',1,'MarkerFaceColor','k')
        end

        box off
        title(datatoplot{n,3})
        xlim([min(x_box_pos)-0.5 max(x_box_pos)+0.5])
        ylim([-5 60])

        mat_data{l,n} = mat_to_plot;
        name_mat_data{l,n} = strcat(strcat(lesion)," ",datatoplot{n,3});
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
        ll_data = size(datatoplot{n,1},1); 
        x_pos = [1,2,2.5,3.5,4,5,5.5];
        x_box_pos = x_pos+((1:numel(x_pos))-median(1:numel(x_pos)))*1/numel(x_pos);
        data_mat_to_plot =  [datatoplot{n,1}(:,2)/meanPreL_total, ...
                            [datatoplot{n,1}(~isnan(datatoplot{n,2}(:,1)),2);NaN(length(datatoplot{n,1})-length(datatoplot{n,1}(~isnan(datatoplot{n,2}(:,1)),2)),1)]/meanPreL_exp,datatoplot{n,2}(:,1)/meanPreL_exp, ...
                            [datatoplot{n,1}(~isnan(datatoplot{n,2}(:,2)),2);NaN(length(datatoplot{n,1})-length(datatoplot{n,1}(~isnan(datatoplot{n,2}(:,2)),2)),1)]/meanPreL_rs,datatoplot{n,2}(:,2)/meanPreL_rs, ...
                            [datatoplot{n,1}(~isnan(datatoplot{n,2}(:,3)),2);NaN(length(datatoplot{n,1})-length(datatoplot{n,1}(~isnan(datatoplot{n,2}(:,3)),2)),1)]/meanPreL_sh,datatoplot{n,2}(:,3)/meanPreL_sh];
        meanMatTot = mean(data_mat_to_plot, 1, 'omitnan');
        x_data_to_plot=ones(size(data_mat_to_plot));
        for colum=1:size(x_data_to_plot,2)
            x_data_to_plot(:,colum)=x_data_to_plot(:,colum)*x_pos(colum);
        end
        % boxchart(reshape(x_data_to_plot,1,[]), reshape(data_mat_to_plot,1,[]),'BoxWidth',3,'MarkerStyle','x','MarkerColor','k','GroupByColor',reshape(x_data_to_plot,1,[]),'BoxFaceAlpha',1,'BoxMedianLineColor','k')
        boxobj = boxchart(reshape(x_data_to_plot,1,[]), reshape(data_mat_to_plot,1,[]),'BoxWidth',3,'MarkerStyle','none','GroupByColor',reshape(x_data_to_plot,1,[]),'BoxFaceAlpha',1,'BoxMedianLineColor','k');
        box_col = [ 77 137 124; 
                    77 137 124; 
                    154 195 232; 
                    77 137 124; 
                    154 195 232; 
                    77 137 124; 
                    154 195 232]/255;
        colororder(box_col)
        xticks(x_box_pos)
        %------------

        xticklabels({})
        hold on
        yline(1,'--')
        for i=1:numel(x_box_pos)
            plot(x_box_pos(i), meanMatTot(i),'Marker','o','MarkerSize',3, 'LineStyle','none','MarkerEdgeColor','k','LineWidth',1,'MarkerFaceColor','k')
        end

        box off
        title(datatoplot{n,3})
        xlim([min(x_box_pos)-0.5 max(x_box_pos)+0.5])
        ylim([-0.1 3.2])

        ratio_mat{l,n} = data_mat_to_plot;
        name_ratio_mat{l,n} = strcat(strcat(lesion)," ",datatoplot{n,3});
        % writematrix(data_mat_to_plot,fullfile(PrePost_PSTHareagraph_boxplot_directory,strcat("ratioPSTHareas ",strcat(lesion)," ",datatoplot{n,3},'.csv')))
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
    
    gr_mem = [];
    for npie=3:8

        currentTile = nexttile(npie);
        
        greater=sum(all_phases(~isnan(all_phases(:,npie-1)),npie-1)>all_mean(ceil(npie/2)))/sum(~isnan(all_phases(:,npie-1)));
        gr_mem = [gr_mem, [greater;1-greater]];

        pie([greater,1-greater]);

        title(all_phases_names(npie-1))

    end 

    nexttile(1:2);
    greater=sum(all_phases(~isnan(all_phases(:,1)),1)>all_mean(1))/sum(~isnan(all_phases(:,1)));
    gr_mem = [gr_mem, [greater;1-greater]];
    pie([greater,1-greater]);
    title(all_phases_names(1))
    legend({'> 1','< 1'})
    legend box off

    title(subp_pie,strcat(lesion," ",datatoplot{n,3}))

    pie_mat{l,n} = gr_mem;
    name_pie_mat{l,n} = strcat(lesion," ",datatoplot{n,3});

    exportgraphics(groups_piechart_PSTHarea_figure,fullfile(PrePost_PSTHareagraph_boxplot_directory,strcat('piePSTHareas_',lesion,'_',datatoplot{n,3},".png")),'Resolution',500)
    groups_piechart_PSTHarea_figure.Visible="on";
    saveas(groups_piechart_PSTHarea_figure,fullfile(PrePost_PSTHareagraph_boxplot_directory,strcat('piePSTHareas_',lesion,'_',datatoplot{n,3},".fig")))
    close(groups_piechart_PSTHarea_figure)

    end

end

%%
%-----------------------------------------------------------
dim = [100 100 700 800];

[val,pos] = max([size(mat_data{1,1},1),size(mat_data{2,1},1)]);

freq1 = cell(4,2);
for q = 1:2
    if pos==1
        freq1{1,q}=[[mat_data{2,q}(:,1:2);NaN(val-size(mat_data{2,q},1),2)], mat_data{1,q}(:,1:2), [ratio_mat{2,q}(:,1);NaN(val-size(mat_data{2,q},1),1)], ratio_mat{1,q}(:,1)];

        freq1{2,q}=[[mat_data{2,q}(:,3:5);NaN(val-size(mat_data{2,q},1),3)], mat_data{1,q}(:,3:5), [ratio_mat{2,q}(:,2:3);NaN(val-size(mat_data{2,q},1),2)], ratio_mat{1,q}(:,2:3)];

        freq1{3,q}=[[mat_data{2,q}(:,6:8);NaN(val-size(mat_data{2,q},1),3)], mat_data{1,q}(:,6:8), [ratio_mat{2,q}(:,4:5);NaN(val-size(mat_data{2,q},1),2)], ratio_mat{1,q}(:,4:5)];

        freq1{4,q}=[[mat_data{2,q}(:,9:11);NaN(val-size(mat_data{2,q},1),3)], mat_data{1,q}(:,9:11), [ratio_mat{2,q}(:,6:7);NaN(val-size(mat_data{2,q},1),2)], ratio_mat{1,q}(:,6:7)];
    else
        freq1{1,q}=[mat_data{2,q}(:,1:2), [mat_data{1,q}(:,1:2);NaN(val-size(mat_data{1,q},1),2)], ratio_mat{2,q}(:,1), [ratio_mat{1,q}(:,1);NaN(val-size(mat_data{1,q},1),1)]];

        freq1{2,q}=[mat_data{2,q}(:,3:5), [mat_data{1,q}(:,3:5);NaN(val-size(mat_data{1,q},1),3)], ratio_mat{2,q}(:,2:3), [ratio_mat{1,q}(:,2:3);NaN(val-size(mat_data{1,q},1),2)]];

        freq1{3,q}=[mat_data{2,q}(:,6:8), [mat_data{1,q}(:,6:8);NaN(val-size(mat_data{1,q},1),3)], ratio_mat{2,q}(:,4:5), [ratio_mat{1,q}(:,4:5);NaN(val-size(mat_data{1,q},1),2)]];

        freq1{4,q}=[mat_data{2,q}(:,9:11), [mat_data{1,q}(:,9:11);NaN(val-size(mat_data{1,q},1),3)], ratio_mat{2,q}(:,6:7), [ratio_mat{1,q}(:,6:7);NaN(val-size(mat_data{1,q},1),2)]];
    end
end

%-------------------------------------------

freq1_imm = figure;
set(gcf,'position',dim)
fin_freq1 = tiledlayout(4,10,"TileSpacing","tight","Padding","tight");
freq1_trans = freq1';
for n=1:8

    %------------
    if n==1
        x_pos1 = [1,1.5,2.5,3];
        x_pos2 = [4,5];
        box_col1 = [77 137 124;
                    154 195 232;
                    77 137 124;
                    154 195 232]/255;
        box_col2 = [68,102,240;
                    68,102,240]/255;
    elseif n==2
        x_pos1 = [1,1.5,2.5,3];
        x_pos2 = [4,5];
        box_col1 = [77 137 124;
                    154 195 232;
                    77 137 124;
                    154 195 232]/255;
        box_col2 = [68,102,240;
                    68,102,240]/255;
    else
        x_pos1 = [1,1.5,2,3,3.5,4];
        x_pos2 = [5,5.5,6.5,7];
        box_col1 = [77 137 124;
                    154 195 232;
                    182 160 78;
                    77 137 124;
                    154 195 232
                    182 160 78]/255;
        box_col2 = [68,102,240;
                    223 104 140;
                    68,102,240;
                    223 104 140]/255;
    end

    x_box_pos1 = x_pos1+((1:numel(x_pos1))-median(1:numel(x_pos1)))*1/numel(x_pos1);
    x_box_pos2 = x_pos2+((1:numel(x_pos2))-median(1:numel(x_pos2)))*1/numel(x_pos2);

    if n==1
        x_tick_pos1 = [mean(x_box_pos1(1:2)), mean(x_box_pos1(3:4))];
        x_tick_pos2 = x_box_pos2;
    elseif n==2
        x_tick_pos1 = [mean(x_box_pos1(1:2)), mean(x_box_pos1(3:4))];
        x_tick_pos2 = x_box_pos2;
    else
        x_tick_pos1 = [mean(x_box_pos1(1:3)), mean(x_box_pos1(4:6))];
        x_tick_pos2 = [mean(x_box_pos2(1:2)), mean(x_box_pos2(3:4))];
    end

    data_mat_to_plot1 =  freq1_trans{n}(:,1:length(x_pos1));
    data_mat_to_plot2 =  freq1_trans{n}(:,(length(x_pos1)+1):end);
    meanMatTot1 = mean(data_mat_to_plot1, 1, 'omitnan');
    meanMatTot2 = mean(data_mat_to_plot2, 1, 'omitnan');
    x_data_to_plot1=ones(size(data_mat_to_plot1));
    for colum=1:size(x_data_to_plot1,2)
        x_data_to_plot1(:,colum)=x_data_to_plot1(:,colum)*x_pos1(colum);
    end
    x_data_to_plot2=ones(size(data_mat_to_plot2));
    for colum=1:size(x_data_to_plot2,2)
        x_data_to_plot2(:,colum)=x_data_to_plot2(:,colum)*x_pos2(colum);
    end

    currentTile = nexttile(((n-1)*5+1),[1 3]);
    boxchart(reshape(x_data_to_plot1,1,[]), reshape(data_mat_to_plot1,1,[]),'BoxWidth',2*1.8*size(x_data_to_plot1,2)/6/2.3*2,'MarkerStyle','none','GroupByColor',reshape(x_data_to_plot1,1,[]),'BoxFaceAlpha',1,'BoxMedianLineColor','k');
    colororder(currentTile, box_col1)
    ylabel('PSTH area')
    hold on
    for i=1:numel(x_box_pos1)
        plot(x_box_pos1(i), meanMatTot1(i),'Marker','o','MarkerSize',3, 'LineStyle','none','MarkerEdgeColor','k','LineWidth',1,'MarkerFaceColor','k')
    end
    xticks(x_tick_pos1)
    xticklabels({'Naïve','Lesioned'})
    box off
    xlim([min(x_box_pos1)-0.5 max(x_box_pos1)+0.5])
    ylim([0 50])

    currentTile = nexttile(((n-1)*5+4),[1 2]);
    boxchart(reshape(x_data_to_plot2,1,[]), reshape(data_mat_to_plot2,1,[]),'BoxWidth',2*1.8*size(x_data_to_plot2,2)/6,'MarkerStyle','none','GroupByColor',reshape(x_data_to_plot2,1,[]),'BoxFaceAlpha',1,'BoxMedianLineColor','k');
    colororder(currentTile, box_col2)
    ylabel('Ratio')
    hold on
    yline(1,'--')
    for i=1:numel(x_box_pos2)
        plot(x_box_pos2(i), meanMatTot2(i),'Marker','o','MarkerSize',3, 'LineStyle','none','MarkerEdgeColor','k','LineWidth',1,'MarkerFaceColor','k')
    end
    xticks(x_tick_pos2)
    xticklabels({'Naïve','Lesioned'})
    box off
    xlim([min(x_box_pos2)-0.5 max(x_box_pos2)+0.5])
    ylim([0 3])

end

xlabel(fin_freq1,'Experimental Group')

n=3;
nexttile(((n-1)*5+1),[1 3]);
legend({'PreL','PoL','PoS'})
nexttile(((n-1)*5+4),[1 2]);
legend({'PoL/PreL','PoS/PreL'},'Location','northwest','FontSize',7.3)

exportgraphics(freq1_imm,fullfile(PrePost_PSTHareagraph_boxplot_directory,strcat("BoxFreq1Hz.png")),'Resolution',500)
saveas(freq1_imm,fullfile(PrePost_PSTHareagraph_boxplot_directory,strcat("BoxFreq1Hz.fig")))
close(freq1_imm)

%-------------------------------------------
%-------------------------------------------

freq0_2 = cell(4,2);
for q = 1:2
    qq=q+2;
    if pos==1
        freq0_2{1,q}=[[mat_data{2,qq}(:,1:2);NaN(val-size(mat_data{2,qq},1),2)], mat_data{1,qq}(:,1:2), [ratio_mat{2,qq}(:,1);NaN(val-size(mat_data{2,qq},1),1)], ratio_mat{1,qq}(:,1)];
    
        freq0_2{2,q}=[[mat_data{2,qq}(:,3:5);NaN(val-size(mat_data{2,qq},1),3)], mat_data{1,qq}(:,3:5), [ratio_mat{2,qq}(:,2:3);NaN(val-size(mat_data{2,qq},1),2)], ratio_mat{1,qq}(:,2:3)];
    
        freq0_2{3,q}=[[mat_data{2,qq}(:,6:8);NaN(val-size(mat_data{2,qq},1),3)], mat_data{1,qq}(:,6:8), [ratio_mat{2,qq}(:,4:5);NaN(val-size(mat_data{2,qq},1),2)], ratio_mat{1,qq}(:,4:5)];

        freq0_2{4,q}=[[mat_data{2,qq}(:,9:11);NaN(val-size(mat_data{2,qq},1),3)], mat_data{1,qq}(:,9:11), [ratio_mat{2,qq}(:,6:7);NaN(val-size(mat_data{2,qq},1),2)], ratio_mat{1,qq}(:,6:7)];
    else
        freq0_2{1,q}=[mat_data{2,qq}(:,1:2), [mat_data{1,qq}(:,1:2);NaN(val-size(mat_data{1,qq},1),2)], ratio_mat{2,qq}(:,1), [ratio_mat{1,qq}(:,1);NaN(val-size(mat_data{1,qq},1),1)]];
    
        freq0_2{2,q}=[mat_data{2,qq}(:,3:5), [mat_data{1,qq}(:,3:5);NaN(val-size(mat_data{1,qq},1),3)], ratio_mat{2,qq}(:,2:3), [ratio_mat{1,qq}(:,2:3);NaN(val-size(mat_data{1,qq},1),2)]];
        
        freq0_2{3,q}=[mat_data{2,qq}(:,6:8), [mat_data{1,qq}(:,6:8);NaN(val-size(mat_data{1,qq},1),3)], ratio_mat{2,qq}(:,4:5), [ratio_mat{1,qq}(:,4:5);NaN(val-size(mat_data{1,qq},1),2)]];

        freq0_2{4,q}=[mat_data{2,qq}(:,9:11), [mat_data{1,qq}(:,9:11);NaN(val-size(mat_data{1,qq},1),3)], ratio_mat{2,qq}(:,6:7), [ratio_mat{1,qq}(:,6:7);NaN(val-size(mat_data{1,qq},1),2)]];
    end
end

%-------------------------------------------

freq2_imm = figure;
set(gcf,'position',dim)
fin_freq2 = tiledlayout(4,10,"TileSpacing","tight","Padding","tight");
freq2_trans = freq0_2';
for n=1:8

    %------------
    if n==1
        x_pos1 = [1,1.5,2.5,3];
        x_pos2 = [4,5];
        box_col1 = [77 137 124;
                    154 195 232;
                    77 137 124;
                    154 195 232]/255;
        box_col2 = [68,102,240;
                    68,102,240]/255;
    elseif n==2
        x_pos1 = [1,1.5,2.5,3];
        x_pos2 = [4,5];
        box_col1 = [77 137 124;
                    154 195 232;
                    77 137 124;
                    154 195 232]/255;
        box_col2 = [68,102,240;
                    68,102,240]/255;
    else
        x_pos1 = [1,1.5,2,3,3.5,4];
        x_pos2 = [5,5.5,6.5,7];
        box_col1 = [77 137 124;
                    154 195 232;
                    182 160 78;
                    77 137 124;
                    154 195 232
                    182 160 78]/255;
        box_col2 = [68,102,240;
                    223 104 140;
                    68,102,240;
                    223 104 140]/255;
    end

    x_box_pos1 = x_pos1+((1:numel(x_pos1))-median(1:numel(x_pos1)))*1/numel(x_pos1);
    x_box_pos2 = x_pos2+((1:numel(x_pos2))-median(1:numel(x_pos2)))*1/numel(x_pos2);

    if n==1
        x_tick_pos1 = [mean(x_box_pos1(1:2)), mean(x_box_pos1(3:4))];
        x_tick_pos2 = x_box_pos2;
    elseif n==2
        x_tick_pos1 = [mean(x_box_pos1(1:2)), mean(x_box_pos1(3:4))];
        x_tick_pos2 = x_box_pos2;
    else
        x_tick_pos1 = [mean(x_box_pos1(1:3)), mean(x_box_pos1(4:6))];
        x_tick_pos2 = [mean(x_box_pos2(1:2)), mean(x_box_pos2(3:4))];
    end

    data_mat_to_plot1 =  freq2_trans{n}(:,1:length(x_pos1));
    data_mat_to_plot2 =  freq2_trans{n}(:,(length(x_pos1)+1):end);
    meanMatTot1 = mean(data_mat_to_plot1, 1, 'omitnan');
    meanMatTot2 = mean(data_mat_to_plot2, 1, 'omitnan');
    x_data_to_plot1=ones(size(data_mat_to_plot1));
    for colum=1:size(x_data_to_plot1,2)
        x_data_to_plot1(:,colum)=x_data_to_plot1(:,colum)*x_pos1(colum);
    end
    x_data_to_plot2=ones(size(data_mat_to_plot2));
    for colum=1:size(x_data_to_plot2,2)
        x_data_to_plot2(:,colum)=x_data_to_plot2(:,colum)*x_pos2(colum);
    end

    currentTile = nexttile(((n-1)*5+1),[1 3]);
    boxchart(reshape(x_data_to_plot1,1,[]), reshape(data_mat_to_plot1,1,[]),'BoxWidth',2*1.8*size(x_data_to_plot1,2)/6/2.3*2,'MarkerStyle','none','GroupByColor',reshape(x_data_to_plot1,1,[]),'BoxFaceAlpha',1,'BoxMedianLineColor','k');
    colororder(currentTile, box_col1)
    ylabel('PSTH area')
    hold on
    for i=1:numel(x_box_pos1)
        plot(x_box_pos1(i), meanMatTot1(i),'Marker','o','MarkerSize',3, 'LineStyle','none','MarkerEdgeColor','k','LineWidth',1,'MarkerFaceColor','k')
    end
    xticks(x_tick_pos1)
    xticklabels({'Naïve','Lesioned'})
    box off
    xlim([min(x_box_pos1)-0.5 max(x_box_pos1)+0.5])
    ylim([0 50])

    currentTile = nexttile(((n-1)*5+4),[1 2]);
    boxchart(reshape(x_data_to_plot2,1,[]), reshape(data_mat_to_plot2,1,[]),'BoxWidth',2*1.8*size(x_data_to_plot2,2)/6,'MarkerStyle','none','GroupByColor',reshape(x_data_to_plot2,1,[]),'BoxFaceAlpha',1,'BoxMedianLineColor','k');
    colororder(currentTile, box_col2)
    ylabel('Ratio')
    hold on
    yline(1,'--')
    for i=1:numel(x_box_pos2)
        plot(x_box_pos2(i), meanMatTot2(i),'Marker','o','MarkerSize',3, 'LineStyle','none','MarkerEdgeColor','k','LineWidth',1,'MarkerFaceColor','k')
    end
    xticks(x_tick_pos2)
    xticklabels({'Naïve','Lesioned'})
    box off
    xlim([min(x_box_pos2)-0.5 max(x_box_pos2)+0.5])
    ylim([0 3])

end

xlabel(fin_freq2,'Experimental Group')

n=3;
nexttile(((n-1)*5+1),[1 3]);
legend({'PreL','PoL','PoS'})
nexttile(((n-1)*5+4),[1 2]);
legend({'PoL/PreL','PoS/PreL'},'Location','northwest','FontSize',7.3)

exportgraphics(freq2_imm,fullfile(PrePost_PSTHareagraph_boxplot_directory,strcat("BoxFreq0.2Hz.png")),'Resolution',500)
saveas(freq2_imm,fullfile(PrePost_PSTHareagraph_boxplot_directory,strcat("BoxFreq0.2Hz.fig")))
close(freq2_imm)

%%
dim = [100 100 700 500];

y=[1,2,4,5, 9,10,12,13, 17,18,20,21, 25,27];
pie_bar_imm1 = figure;
set(gcf,'position',dim)
pie1 = tiledlayout(1,2,"TileSpacing","tight","Padding","tight");
for i=1:2
    x = [pie_mat{1,i}(:,end-1:-1:end-2),pie_mat{2,i}(:,end-1:-1:end-2),...
        pie_mat{1,i}(:,end-3:-1:end-4),pie_mat{2,i}(:,end-3:-1:end-4),...
        pie_mat{1,i}(:,end-5:-1:end-6),pie_mat{2,i}(:,end-5:-1:end-6),...
        pie_mat{1,i}(:,end),pie_mat{2,i}(:,end)];
    
    nexttile(i, [1 1])
    bb1 = barh(y(:,[2:2:12,13:14]),x(:,[2:2:12,13:14])'*100,'stacked');
    xlim([0,100])
    bb1(1).FaceColor = [68,102,240]/255;
    bb1(2).FaceColor = [68,102,240]/255;
    bb1(2).FaceAlpha = 0.5;
    bb1(1).BarWidth = 0.4;

    hold on
    bb2 = barh(y(:,1:2:12),x(:,1:2:12)'*100,'stacked');
    bb2(1).FaceColor = [223 104 140]/255;
    bb2(2).FaceColor = [223 104 140]/255;
    bb2(1).BarWidth = 0.25;
    bb2(2).FaceAlpha = 0.5;

    % ytips1 = bb1(1).XEndPoints;
    % xtips1 = bb1(1).YEndPoints/2;
    % labels1 = string(round(bb1(1).YData,0));
    % text(xtips1,ytips1,labels1,'VerticalAlignment','middle','HorizontalAlignment','center')
    %
    % ytips2 = bb2(1).XEndPoints;
    % xtips2 = bb2(1).YEndPoints/2;
    % labels2 = string(round(bb2(1).YData,0));
    % text(xtips2,ytips2,labels2,'VerticalAlignment','middle','HorizontalAlignment','center')

    % xline(50,'--','LineWidth',0.2,'Color',[0.45 0.45 0.45])
    set(gca,'XGrid','on','YGrid','off')
    xticks(0:10:100)
    % xticklabels({'','','','','','50','','','','',''})
    % legend({'PoL incr','PoL decr','PoS incr','PoS decr'},'Location','northoutside')
    yticks([1.5 4.5 9.5 12.5 17.5 20.5 25 27])
    yticklabels({'Lesioned','Naïve','Lesioned','Naïve','Lesioned','Naïve','Lesioned','Naïve',})
end

xlabel(pie1,'Percentage [%]')

nexttile(2,[1 1]);
legend({'PoL incr','PoL decr','PoS incr','PoS decr'},'Location','northoutside')

exportgraphics(pie_bar_imm1,fullfile(PrePost_PSTHareagraph_boxplot_directory,strcat("BarPieFreq1Hz.png")),'Resolution',500)
saveas(pie_bar_imm1,fullfile(PrePost_PSTHareagraph_boxplot_directory,strcat("BarPieFreq1Hz.fig")))
close(pie_bar_imm1)
%--------------------------------

pie_bar_imm2 = figure;
set(gcf,'position',dim)
pie2 = tiledlayout(1,2,"TileSpacing","tight","Padding","tight");
for i=1:2
    ii=i+2;
    x = [pie_mat{1,ii}(:,end-1:-1:end-2),pie_mat{2,ii}(:,end-1:-1:end-2),...
        pie_mat{1,ii}(:,end-3:-1:end-4),pie_mat{2,ii}(:,end-3:-1:end-4),...
        pie_mat{1,ii}(:,end-5:-1:end-6),pie_mat{2,ii}(:,end-5:-1:end-6),...
        pie_mat{1,ii}(:,end),pie_mat{2,ii}(:,end)];
    
    nexttile(i, [1 1])
    bb1 = barh(y(:,[2:2:12,13:14]),x(:,[2:2:12,13:14])'*100,'stacked');
    xlim([0,100])
    bb1(1).FaceColor = [68,102,240]/255;
    bb1(2).FaceColor = [68,102,240]/255;
    bb1(2).FaceAlpha = 0.5;
    bb1(1).BarWidth = 0.4;

    hold on
    bb2 = barh(y(:,1:2:12),x(:,1:2:12)'*100,'stacked');
    bb2(1).FaceColor = [223 104 140]/255;
    bb2(2).FaceColor = [223 104 140]/255;
    bb2(1).BarWidth = 0.25;
    bb2(2).FaceAlpha = 0.5;

    % xline(50,'--','LineWidth',0.2,'Color',[0.45 0.45 0.45])
    set(gca,'XGrid','on','YGrid','off')
    xticks(0:10:100)
    % xticklabels({'','','','','','50','','','','',''})
    % legend({'PoL incr','PoL decr','PoS incr','PoS decr'},'Location','northoutside')
    yticks([1.5 4.5 9.5 12.5 17.5 20.5 25 27])
    yticklabels({'Lesioned','Naïve','Lesioned','Naïve','Lesioned','Naïve','Lesioned','Naïve',})
end

xlabel(pie2,'Percentage [%]')

nexttile(2,[1 1]);
legend({'PoL incr','PoL decr','PoS incr','PoS decr'},'Location','northoutside')

exportgraphics(pie_bar_imm2,fullfile(PrePost_PSTHareagraph_boxplot_directory,strcat("BarPieFreq0.2Hz.png")),'Resolution',500)
saveas(pie_bar_imm2,fullfile(PrePost_PSTHareagraph_boxplot_directory,strcat("BarPieFreq0.2Hz.fig")))
close(pie_bar_imm2)
%%
% 1Hz

stat_4_test_1Hz = cell(4,2);
for i=1:2

    %All
    if size(statistical_test_matrix{2,1}{i,1},1)>size(statistical_test_matrix{1,1}{i,1},1)
        stat_4_test_1Hz{1,i}=[statistical_test_matrix{2,1}{i,1}, [statistical_test_matrix{1,1}{i,1};NaN(abs(size(statistical_test_matrix{1,1}{i,1},1)-size(statistical_test_matrix{2,1}{i,1},1)),size(statistical_test_matrix{2,1}{i,1},2))]];
    else
        stat_4_test_1Hz{1,i}=[[statistical_test_matrix{2,1}{i,1};NaN(abs(size(statistical_test_matrix{2,1}{i,1},1)-size(statistical_test_matrix{1,1}{i,1},1)),size(statistical_test_matrix{2,1}{i,1},2))], statistical_test_matrix{1,1}{i,1}];
    end

    %Exp
    k=1;
    if size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)>size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)
        stat_4_test_1Hz{2,i}=[statistical_test_matrix{2,1}{i,1}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),1:2), statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k)...
                             [statistical_test_matrix{1,1}{i,1}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),1:2);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),size(statistical_test_matrix{1,1}{i,1},2))],...
                             [statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),1)]];
    else
        stat_4_test_1Hz{2,i}=[[statistical_test_matrix{2,1}{i,1}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),1:2);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),size(statistical_test_matrix{1,1}{i,1},2))],...
                             [statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),1)],...
                             statistical_test_matrix{1,1}{i,1}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),1:2),statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k)];
    end

    %Rs
    k=2;
    if size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)>size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)
        stat_4_test_1Hz{3,i}=[statistical_test_matrix{2,1}{i,1}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),1:2), statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k)...
                             [statistical_test_matrix{1,1}{i,1}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),1:2);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),size(statistical_test_matrix{1,1}{i,1},2))],...
                             [statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),1)]];
    else
        stat_4_test_1Hz{3,i}=[[statistical_test_matrix{2,1}{i,1}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),1:2);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),size(statistical_test_matrix{1,1}{i,1},2))],...
                             [statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),1)],...
                             statistical_test_matrix{1,1}{i,1}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),1:2),statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k)];
    end

    %Sh
    k=3;
    if size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)>size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)
        stat_4_test_1Hz{4,i}=[statistical_test_matrix{2,1}{i,1}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),1:2), statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k)...
                             [statistical_test_matrix{1,1}{i,1}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),1:2);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),size(statistical_test_matrix{1,1}{i,1},2))],...
                             [statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),1)]];
    else
        stat_4_test_1Hz{4,i}=[[statistical_test_matrix{2,1}{i,1}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),1:2);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),size(statistical_test_matrix{1,1}{i,1},2))],...
                             [statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),1)],...
                             statistical_test_matrix{1,1}{i,1}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),1:2),statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k)];
    end

end


% 0.2Hz

stat_4_test_02Hz = cell(4,2);
for i=3:4

    %All
    if size(statistical_test_matrix{2,1}{i,1},1)>size(statistical_test_matrix{1,1}{i,1},1)
        stat_4_test_02Hz{1,i-2}=[statistical_test_matrix{2,1}{i,1}, [statistical_test_matrix{1,1}{i,1};NaN(abs(size(statistical_test_matrix{1,1}{i,1},1)-size(statistical_test_matrix{2,1}{i,1},1)),size(statistical_test_matrix{2,1}{i,1},2))]];
    else
        stat_4_test_02Hz{1,i-2}=[[statistical_test_matrix{2,1}{i,1};NaN(abs(size(statistical_test_matrix{2,1}{i,1},1)-size(statistical_test_matrix{1,1}{i,1},1)),size(statistical_test_matrix{2,1}{i,1},2))], statistical_test_matrix{1,1}{i,1}];
    end

    %Exp
    k=1;
    if size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)>size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)
        stat_4_test_02Hz{2,i-2}=[statistical_test_matrix{2,1}{i,1}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),1:2), statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k)...
                             [statistical_test_matrix{1,1}{i,1}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),1:2);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),size(statistical_test_matrix{1,1}{i,1},2))],...
                             [statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),1)]];
    else
        stat_4_test_02Hz{2,i-2}=[[statistical_test_matrix{2,1}{i,1}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),1:2);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),size(statistical_test_matrix{1,1}{i,1},2))],...
                             [statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),1)],...
                             statistical_test_matrix{1,1}{i,1}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),1:2),statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k)];
    end

    %Rs
    k=2;
    if size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)>size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)
        stat_4_test_02Hz{3,i-2}=[statistical_test_matrix{2,1}{i,1}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),1:2), statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k)...
                             [statistical_test_matrix{1,1}{i,1}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),1:2);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),size(statistical_test_matrix{1,1}{i,1},2))],...
                             [statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),1)]];
    else
        stat_4_test_02Hz{3,i-2}=[[statistical_test_matrix{2,1}{i,1}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),1:2);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),size(statistical_test_matrix{1,1}{i,1},2))],...
                             [statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),1)],...
                             statistical_test_matrix{1,1}{i,1}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),1:2),statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k)];
    end

    %Sh
    k=3;
    if size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)>size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)
        stat_4_test_02Hz{4,i-2}=[statistical_test_matrix{2,1}{i,1}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),1:2), statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k)...
                             [statistical_test_matrix{1,1}{i,1}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),1:2);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),size(statistical_test_matrix{1,1}{i,1},2))],...
                             [statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),1)]];
    else
        stat_4_test_02Hz{4,i-2}=[[statistical_test_matrix{2,1}{i,1}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),1:2);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),size(statistical_test_matrix{1,1}{i,1},2))],...
                             [statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k);NaN(abs(size(statistical_test_matrix{2,1}{i,2}(~isnan(statistical_test_matrix{2,1}{i,2}(:,k)),k),1)-size(statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k),1)),1)],...
                             statistical_test_matrix{1,1}{i,1}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),1:2),statistical_test_matrix{1,1}{i,2}(~isnan(statistical_test_matrix{1,1}{i,2}(:,k)),k)];
    end

end
%%
riga = 1;
colonna = 1;

% 1Hz
[p,~,stats] = anova1(stat_4_test_1Hz{riga,colonna});
results = multcompare(stats);
tbl = array2table(results,"VariableNames", ...
    ["Group A","Group B","Lower Limit","A-B","Upper Limit","P-value"]);
disp(tbl)

% 0.2Hz
[p,~,stats] = anova1(stat_4_test_02Hz{riga,colonna});
results = multcompare(stats);
tbl = array2table(results,"VariableNames", ...
    ["Group A","Group B","Lower Limit","A-B","Upper Limit","P-value"]);
disp(tbl)