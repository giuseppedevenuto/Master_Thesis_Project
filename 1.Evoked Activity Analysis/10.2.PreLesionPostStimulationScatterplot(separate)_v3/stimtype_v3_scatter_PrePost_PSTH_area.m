% plotta solo pre-lesione contro post-stimolazione
% nei lesionati si spera di ottenere una bisettrice

% in una stessa figura mette tutti gli scatter di tutti i gruppi
% considerando anche le diverse tipologie di stimolazioni ol 
% --> crea un subplot 4x8 (nel nostro caso)

clear
clc

nareas_x_nmapping=4;
ncolumn = 3;
nphasemax = 3;
nstimphase = 2;
axis_max_lim=50;

PrePost_PSTHarea_directory=uigetdir('C:\Users\Giuseppe\Desktop','Select Directory Where To Load (la cartella coi gruppi di ratti su cui lavorare)');
StimType_PrePost_PSTHareagraph_scatter_directory=uigetdir(fullfile(PrePost_PSTHarea_directory,'\..'),'Select Directory Where To Save (la cartella in cui salvare gli scatterplot delle aree dei PSTH per tipo di stim)');

groups = dir(fullfile(PrePost_PSTHarea_directory,'*-*'));
groupnames = {groups.name};
clear groups
ngroups = length(groupnames);

groups_scatter_PSTHarea_figure=figure(Visible="on",WindowState="maximized");

% subp_scatter = tiledlayout(ngroups,nareas_x_nmapping,"TileSpacing","tight","Padding","tight");
% pos_graph = reshape(1:nareas_x_nmapping*ngroups,[ngroups,nareas_x_nmapping]);
subp_scatter = tiledlayout(nareas_x_nmapping,ngroups,"TileSpacing","tight","Padding","tight");
pos_graph = reshape(1:nareas_x_nmapping*ngroups,[ngroups,nareas_x_nmapping])';

for p=1:ngroups

    rats = dir (fullfile(PrePost_PSTHarea_directory,groupnames{p},'R*-*'));
    ratnames={rats.name};
    clear rats
    nrats=length(ratnames);

    datatoplot = cell(nareas_x_nmapping,ncolumn);
   
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

                    temp=zeros(size(electrodesPSTHareas_NaN,1),nphasemax);
                    for phase=1:length(mapphases)
                       
                        if str2double(mapphases{phase}(2))==nstimphase+1
                            col_phase=str2double(mapphases{phase}(2));
                        else
                            col_phase=str2double(mapphases{phase}(2))+1;
                        end
                        temp(:,col_phase) = electrodesPSTHareas_NaN(:,phase);
                       
                    end

                    datatoplot{row,1}(end+1:end+size(electrodesPSTHareas_NaN,1),:) = temp;
                    datatoplot{row,2} = strcat(area1," ",maptypenames{h});
                    datatoplot{row,3} = mapphases;

                end
            end
        end   
    end

    legendtoplot=cell(1,1);
    colours={[0, 0.4470, 0.7410],[0.8500, 0.3250, 0.0980],[0.4940, 0.1840, 0.5560]};

    for n=1:nareas_x_nmapping
        datatoplot{n,1}(datatoplot{n,1}==0) = NaN;
        datatoplot{n,1}(:,~any(datatoplot{n,1}))=[];
        if length(datatoplot{n,3})==3
        datatoplot{n,1}(:,2)=[];
        datatoplot{n,3}(2,:)=[];
        end
        % currentTile = nexttile((p-1)*nareas_x_nmapping+n);
        currentTile = nexttile(pos_graph((p-1)*nareas_x_nmapping+n));

        for w=1:length(datatoplot{n,3})-1

            Non_NaN_idx = and(any(datatoplot{n,1}(:,w),2),any(datatoplot{n,1}(:,w+1),2));

            line(datatoplot{n,1}(Non_NaN_idx,w),datatoplot{n,1}(Non_NaN_idx,w+1),'linestyle','none','marker','.','MarkerSize',10,"Color", [0; 0.45; 0.74])
            hold on
            slope=datatoplot{n,1}(Non_NaN_idx,w)\datatoplot{n,1}(Non_NaN_idx,w+1);
            regression=slope*[0; datatoplot{n,1}(Non_NaN_idx,w); axis_max_lim];
            plot([0; datatoplot{n,1}(Non_NaN_idx,w); axis_max_lim],regression,Color='r',LineWidth=1)
            
            legendtoplot{1}(2*w-1)=strcat(datatoplot{n,3}{w}(1),datatoplot{n,3}{w}(2),{' VS '},datatoplot{n,3}{w+1}(1),datatoplot{n,3}{w+1}(2)," (",num2str(slope,'%.4f'),")");
            legendtoplot{1}(2*w)=strcat({''});
        end

        plot([0 axis_max_lim],[0 axis_max_lim],'k','LineStyle','--')
        legendtoplot{1}{2*length(datatoplot{n,3})-1}=strcat('slope = 1');
        lgd = legend(legendtoplot{1},"Location","northwest");
        lgd.FontSize = 5;
        title(lgd,strcat(groupnames{p}," ",datatoplot{n,2}),"FontSize",6)
        legend box off
        box off
        xlim([0 axis_max_lim])
        ylim([0 axis_max_lim])

    end
end

ylabel(subp_scatter,'PSTH areas Post')
xlabel(subp_scatter,'PSTH areas Pre')
exportgraphics(groups_scatter_PSTHarea_figure,fullfile(StimType_PrePost_PSTHareagraph_scatter_directory,"PSTHarea_orizzontale_inizio_fine_esperimento.png"),'Resolution',500)
groups_scatter_PSTHarea_figure.Visible="on";
saveas(groups_scatter_PSTHarea_figure,fullfile(StimType_PrePost_PSTHareagraph_scatter_directory,"PSTHareas_orizzontale_inizio_fine_esperimento.fig"))
close(groups_scatter_PSTHarea_figure)