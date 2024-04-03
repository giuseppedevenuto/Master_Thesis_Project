% creazione dei plot dei psth per ogni singolo ratto discriminando tipo
% di mapping, canale di stimolazione e area. Creazione dei
% barplot e scatterplot delle aree dei dei psth dei ratti discriminando 
% canale di stimolazione e area e tipo di mapping
% salvataggio delle aree dei psth per le ulteriori analisi di gruppo
% (divisi in sham-sham, sham-ol-exp, sham-ol-sh, sham-ol-rs, sham-cl, 
% et1-ol-exp, et1-ol-sh, et1-ol-rs)

clear
clc

fs=25000;
wdwsize_time = 0.8; % sec - wdw after stimulus
binsize_time = 0.004; % sec
artremoval_wdwsize_time = 0.0004; % sec
y_lim=200;

ele_array = [1, 7, 13, 14;
             3, 4, 10, 16;
             2, 8, 12, 11; 
             6, 5, 9, 15];

PrePost_PSTH_directory=uigetdir('C:\Users\Giuseppe\Desktop','Select Directory Where To Save (la cartella coi PSTH del gruppo di ratti su cui lavorare)');
PrePost_PSTHgraph_directory=uigetdir(fullfile(PrePost_PSTH_directory,'\..\..'),'Select Directory Where To Save (la cartella col nome del gruppo di ratti in cui salvare i PSTH)');
PrePost_PSTHareagraph_barplot_directory=uigetdir(fullfile(PrePost_PSTH_directory,'\..\..'),'Select Directory Where To Save (la cartella col nome del gruppo di ratti in cui salvare i barplot delle aree dei PSTH)');
PrePost_PSTHareagraph_scatter_directory=uigetdir(fullfile(PrePost_PSTH_directory,'\..\..'),'Select Directory Where To Save (la cartella col nome del gruppo di ratti in cui salvare gli scatterplot delle aree dei PSTH)');
PrePost_PSTHarea_directory=uigetdir(fullfile(PrePost_PSTH_directory,'\..\..'),'Select Directory Where To Save (la cartella in cui salvare le matrici con le aree dei PSTH dei vari gruppi di ratti)');
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

            %
            for r=1:size(rats_groups_table,1)
                if ~any(rats_groups_table{r,2}~=ratnames{k})
                    idx_rat = r;
                end
            end
            mkdir(fullfile(PrePost_PSTHarea_directory,rats_groups_table{idx_rat,1},ratnames{k},mapping,area1))
            %

            for g=1:nhalfstims
                electrodes = dir(fullfile(PrePost_PSTH_directory,ratnames{k},maptypenames{h},area1,halfstimsnames{g},'*.mat'));
                electrodenames = {electrodes.name};
                PSTH_electrodes_directory = electrodes.folder;
                clear electrodes
                nelectrodes = length(electrodenames);
                
                X=(0:wdwsize_time/binsize_time-1)*0.004;
                electrodesPSTHareas=zeros(nelectrodes,1);

                electrodesPSTHfigure=figure(Visible="off");

                subp_elec = tiledlayout(ceil(sqrt(nelectrodes)),ceil(sqrt(nelectrodes)),"TileSpacing","tight","Padding","tight");
                               
                for n=1:nelectrodes
                    load(fullfile(PSTH_electrodes_directory,electrodenames{n}));

                    electrodePSTHareas=zeros(1,length(PrePostPSTH{1}));

                    currentTile = nexttile(find(ele_array'==n));
                    
                    time = 0:binsize_time:(wdwsize_time-binsize_time);
                    for m=1:length(PrePostPSTH{1})

                        electrodePSTHareas(m)=trapz(X, PrePostPSTH{2}(m,:));

                        if electrodePSTHareas(m) ~= 0
                            plot(time,PrePostPSTH{2}(m,:))
                            hold on
                        else
                            PrePostPSTH{2}(m,PrePostPSTH{2}(m,:)==0) = NaN;
                            plot(time,PrePostPSTH{2}(m,:))
                            hold on
                        end
                    end

                    electrodesPSTHareas(n,1:length(PrePostPSTH{1}))=electrodePSTHareas;

                    hold off
                    box off
                    xticks(0:(wdwsize_time/2):wdwsize_time)
                    xlim([0 wdwsize_time])
                    ylim([0 y_lim])
                    title(strcat('Channel',num2str(n,'%02.f')))


                    if t == 2 && n == (where_stim_ch-16)

                        box on

                        currentTile.LineWidth = 1;
                        currentTile.XColor = [1 0 0];
                        currentTile.YColor = [1 0 0];
                    end
                end

                currentTile = nexttile(1);
                lgd = legend(PrePostPSTH{1});
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

                exportgraphics(electrodesPSTHfigure,fullfile(PrePost_PSTHgraph_directory,strcat('PSTH_',ratnames{k},"_",Stims{1},"_",maptypenames{h},"_",area1,".emf")),'ContentType','vector')
                exportgraphics(electrodesPSTHfigure,fullfile(PrePost_PSTHgraph_directory,strcat('PSTH_',ratnames{k},"_",Stims{1},"_",maptypenames{h},"_",area1,".png")),'Resolution',500)
                electrodesPSTHfigure.Visible="on";
                saveas(electrodesPSTHfigure,fullfile(PrePost_PSTHgraph_directory,strcat('PSTH_',ratnames{k},"_",Stims{1},"_",maptypenames{h},"_",area1,".fig")))
                close(electrodesPSTHfigure)

                electrodesPSTHareasfigure=figure(Visible="off");
                bar(electrodesPSTHareas)
                xticks(1:16)
                legend(PrePostPSTH{1},"Location","northeastoutside")
                legend boxoff
                box off
                xlabel('channels [#]')
                ylabel('PSTH areas')

                title(strcat("Rat-",Rats{1}," ",ch_stim," ",mapping," ",area1))
                exportgraphics(electrodesPSTHareasfigure,fullfile(PrePost_PSTHareagraph_barplot_directory,strcat('PSTHareas_',ratnames{k},"_",Stims{1},"_",maptypenames{h},"_",area1,".png")),'Resolution',500)
                
                electrodesPSTHareas_NaN = electrodesPSTHareas;
                electrodesPSTHareas_NaN(electrodesPSTHareas_NaN==0) = NaN;
                legendtoplot=cell(1,1);
                colours={[0, 0.4470, 0.7410],[0.8500, 0.3250, 0.0980],[0.4940, 0.1840, 0.5560]};
                
                scatterPSTHareasfigure=figure(Visible="off");
                % figure
                for w=1:length(PrePostPSTH{1})-1

                    Non_NaN_idx = and(any(electrodesPSTHareas_NaN(:,w),2),any(electrodesPSTHareas_NaN(:,w+1),2));

                    line(electrodesPSTHareas_NaN(Non_NaN_idx,w),electrodesPSTHareas_NaN(Non_NaN_idx,w+1),'linestyle','none','marker','.','MarkerSize',10,'Color',colours{w}(:))
                    hold on
                    slope=electrodesPSTHareas_NaN(Non_NaN_idx,w)\electrodesPSTHareas_NaN(Non_NaN_idx,w+1);
                    regression=slope*[0; electrodesPSTHareas_NaN(Non_NaN_idx,w)];
                    plot([0; electrodesPSTHareas_NaN(Non_NaN_idx,w)],regression,Color=colours{w}(:))

                    legendtoplot{1}(2*w-1)=strcat(PrePostPSTH{1}{w}(1),PrePostPSTH{1}{w}(2),{' VS '},PrePostPSTH{1}{w+1}(1),PrePostPSTH{1}{w+1}(2));
                    legendtoplot{1}(2*w)=strcat({''});

                end
                plot([0 max(electrodesPSTHareas_NaN,[],"all")],[0 max(electrodesPSTHareas_NaN,[],"all")],'k','LineStyle','--')
                legendtoplot{1}{2*length(PrePostPSTH{1})-1}=strcat('slope = 1');
                legend(legendtoplot{1},"Location","northeastoutside")
                box off
                xlabel('PSTH areas Pre')
                ylabel('PSTH areas Post')

                title(strcat("Rat-",Rats{1}," ",ch_stim," ",mapping," ",area1))
                exportgraphics(scatterPSTHareasfigure,fullfile(PrePost_PSTHareagraph_scatter_directory,strcat('PSTHareas_',ratnames{k},"_",Stims{1},"_",maptypenames{h},"_",area1,".png")),'Resolution',500)

                mapphases=PrePostPSTH{1};
                save(fullfile(PrePost_PSTHarea_directory,rats_groups_table{idx_rat,1},ratnames{k},mapping,area1,strcat('PrePostPSTHarea_',ch_stim)),'electrodesPSTHareas_NaN','mapphases')

            end
        end
    end
end