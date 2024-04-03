# -*- coding: utf-8 -*-
# @title      Generate configuration for organoid emulation
# @file       RatBrain.py
# @author     Romain Beaubois
# @date       17 Feb 2022
# @copyright
# SPDX-FileCopyrightText: © 2022 Romain Beaubois <refbeaubois@yahoo.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# @brief Generate configuration for organoid emulation
# This package only generates configuration parameters for SNN-HH on FPGA.
# It has to be emulated by SnnEmulator from the package file generated.
# 
# @details 
# > **17 Feb 2022** : file creation (RB)

import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
import pandas as pd

## Simulation parameters ##################################################
# Describe the models of neurons and synapses used to model the organoid

# Neuron model
NRN_EXC     = 0
NRN_INH     = 1

nrncode     = { NRN_EXC: "RSrat",   # Excitatory neuron is customized RS from "HHparam.py" <EDIT>
                NRN_INH: "FSorg"}   # Inhibitory neuron is customized FS from "HHparam.py" <EDIT>

# Synapses model
SYN_NONE        = 0
SYN_EXC_AMPA    = 1
SYN_EXC_NMDA    = 2
SYN_INH         = 3

syncode     = { SYN_NONE:       "destexhe_none",      # No synapse is none from "Synapses.py"
                SYN_EXC_AMPA:   "destexhe_ampa",      # Excitatory synapse is AMPA from "Synapses.py"
                SYN_EXC_NMDA:   "destexhe_nmda",      # Excitatory synapse is NMDA from "Synapses.py"
                SYN_INH:        "destexhe_gabaa"}     # Inhibitory synapse is GABAa from "Synapses.py"

# Colors
COLOR_EXC = "red"   # Color for excitatory neurons and synaptic connections
COLOR_INH = "blue"  # Color for inhibitory neurons and synaptic connections

## Organoid modeling ##################################################
# Helper class to generate configuration of organoid modeling
class RatBrain:
    def __init__(self, nb_nrn) -> None:
        """Initialize
        
        :param int nb_nrn: Number of neurons used to emulate the WHOLE model
        """
        self.nb_nrn         = nb_nrn    # Number of neurons used in total
        self.nb_orgs        = 0         # Number of organoids
        self.nb_nrn_per_org = 0         # Number of neurons per organoid
        self.org_diam       = []        # List of organoids diameters
        self.nrn_diam       = []        # List of neuron diameters per organoid
        self.org_center_xy  = []        # List of XY coordinate for oragnoids centers

        self.x          = [] # List of neuron's x coordinates [org_i, n_j]
        self.y          = [] # List of neuron's y coordinates [org_i, n_j]
        self.dist2c     = [] # List of neuron's distance to center [org_i, n_j]
        self.nlist      = [] # List of neuron's index [org_i, n_j]
        self.tnrn       = [] # List of neuron's types [org_i, n_j]
        self.x_all      = [] # List of neuron's x coordinates of all neurons [n0 ... self.nb_nrn]
        self.y_all      = [] # List of neuron's y coordinates of all neurons [n0 ... self.nb_nrn]
        self.dist2c_all = [] # List of neuron's distance to center of all neurons [n0 ... self.nb_nrn]

        self.tnrn_all   = [] # List of neuron's type of all neurons [n0 ... self.nb_nrn]
        self.nlist_all  = [] # List of neuron's index of all neurons [n0 ... self.nb_nrn]
        self.tsyn       = np.zeros([self.nb_nrn, self.nb_nrn], dtype=np.uint8)  # Synapses types [dest, src]
        self.wsyn       = np.zeros([self.nb_nrn, self.nb_nrn])  # Synapses types [dest, src]

    # Setters -------------------------------------------------------------------------

    def addOrganoid(self, org_diam, nrn_diam, org_center_xy):
        """Add an organoid to modeling
        
        :param int org_diam: Diameter of organoid
        :param int nrn_diam: Diameter of neurons
        :param list org_center_xy: XY coordinates of organoid
        """
        self.nb_orgs += 1
        self.nb_nrn_per_org = int(self.nb_nrn/self.nb_orgs)
        self.org_diam.append(org_diam)
        self.nrn_diam.append(nrn_diam)
        self.org_center_xy.append(org_center_xy)

    def genNeurons(self, inh_ratio):
        """Generate neurons' XY coordinates and types
        
        :param float inh_ratio: Ratio of inhibitory neurons per organoid
        """
        self.inh_ratio      = inh_ratio

        # For all organoid
        for i in range(self.nb_orgs):
            # Generate neuron index
            self.nlist.append([ n for n in range(i*self.nb_nrn_per_org, (i+1)*self.nb_nrn_per_org) ])

            # Generate XY coordignates
            [xt, yt, dist2ct] = (self.__genXYcoordinates("smallworld", self.org_diam[i], self.org_center_xy[i][:], self.nb_nrn_per_org)) # <EDIT> 
            self.x.append(xt)
            self.y.append(yt)
            self.dist2c.append(dist2ct)

            # Generate neuron types
            self.tnrn.append(self.__genNrnTypes(self.nb_nrn_per_org, self.inh_ratio))

        # Create list shaped based on neurons (snn-hh just consider neurons connected)
        self.x_all      =  [item for sublist in self.x      for item in sublist]
        self.y_all      =  [item for sublist in self.y      for item in sublist]
        self.dist2c_all =  [item for sublist in self.dist2c for item in sublist]
        self.tnrn_all   =  [item for sublist in self.tnrn   for item in sublist]
        self.nlist_all  =  [item for sublist in self.nlist  for item in sublist]

    def genSynCon(self, rule, org_dest, org_src, max_pcon, config_param):
        """Generate synaptic connections
        
        :param str rule: Rule applied for synaptic connection ("single", "assembloid", "connectoid")
        :param int org_dest: Index of destination organoid
        :param int org_src: Index of source organoid
        :param float max_pcon: Maximum connection probability

        "single": connection among one organoid based on distance between the neurons to connect (linear)
        "assembloid": connection between organoids based on distance between the neurons to connect (linear or exponential)
        "connectoid": connection between organoids based on positions of organoid in the organoids (linear)

        Examples: genSynCon("single", org_dest=0, org_src=0, 0.1); genSynCon("assembloid", org_dest=0, org_src=1, 0.2)
        """
        if   rule == "single":
            self.tsyn = self.__ruleSynConSingle(    self.tsyn, self.nlist[org_dest], self.nlist[org_src], self.tnrn_all, self.x_all, self.y_all, self.org_diam[org_src], max_pcon)
        elif rule == "assembloid":
            dist_orgs = math.sqrt(  (self.org_center_xy[org_dest][0]-self.org_center_xy[org_src][0])**2 + 
                                    (self.org_center_xy[org_dest][1]-self.org_center_xy[org_src][1])**2
                                 )
            self.tsyn = self.__ruleSynConAssembloid(self.tsyn, self.nlist[org_dest], self.nlist[org_src], self.tnrn_all, self.x_all, self.y_all, dist_orgs + self.org_diam[org_src]/2+self.org_diam[org_dest]/2, max_pcon)
        elif rule == "connectoid":
            self.tsyn = self.__ruleSynConConnectoid(self.tsyn, self.nlist[org_dest], self.nlist[org_src], self.tnrn_all, self.x_all, self.y_all, self.dist2c_all, max_pcon)
        elif rule == "smallworld":
            self.tsyn = self.__ruleSynConSmallWorld(self.tsyn, self.nlist[org_dest], self.nlist[org_src], self.tnrn_all, self.x_all, self.y_all, self.org_diam[org_src], max_pcon, config_param = config_param)

    def genSynWeights(self, org_dest=-1, org_src=-1, weight=0.0):
        """Generate synaptic weights
        
        :param int org_dest: Index of destination organoid
        :param int org_src: Index of source organoid
        :param float weight: Synaptic weight to apply

        Passing -1 for both organoid index select all synapses.
        For now the same weight is apply to synapses in the group.

        Example: genSynWeights(org_dest=-1, org_src=-1, weight=10.0) sets all synapses to 10.0
        """
        if (org_src != -1) and (org_dest != -1):
            self.wsyn = self.__genGroupSynWeights(self.nlist[org_dest], self.nlist[org_src], self.wsyn, weight)
        else:
            self.wsyn.fill(weight)

    def __genXYcoordinates(self, type:str, diam, center_xy, nb_nrn=0, nrn_diam=0.0):
        """Generate XY coordinates
        
        :param str type: Type of XY repartition ("random", "radius")
        :param float diam: Diameter of the disk
        :param list center_xy: Coordinates of the disk center
        :param int nb_nrn: Number of neurons to place in the disk
        :param float nrn_diam: Diameter of neurons
        """
        x       = []
        y       = []
        dist2c  = []

        if   type == "random":
            for _ in range(nb_nrn):
                u       = np.random.rand() + np.random.rand()
                alpha   = 2 * math.pi * np.random.rand()
                # r       = (diam/2) * math.sqrt(np.random.rand())
                r       = (diam/2) * (2 - u if u > 1 else u)
                xt      = r * math.cos(alpha) + center_xy[0]
                yt      = r * math.sin(alpha) + center_xy[1]
                dc      = math.sqrt((center_xy[0]-xt)**2 + (center_xy[1]-yt)**2)/(diam/2)

                x.append( xt )
                y.append( yt )
                dist2c.append( dc )

        elif type == "radius":
            ring_diam   = diam
            delta_a     = nrn_diam*180/(np.pi*(ring_diam/2))
            keepout     = nrn_diam*1.25
            
            ## Neuron at the middle of the cluster
            x = []
            y = []
            x.append(center_xy[0])
            y.append(center_xy[1])

            ## Calculate coordinates
            cnt_nrn = 1
            while 1:
                theta = 0
                while theta < (360-delta_a):
                    x_t = center_xy[0] + (ring_diam/2) * math.cos(theta*np.pi/180)
                    y_t = center_xy[1] + (ring_diam/2) * math.sin(theta*np.pi/180)
                    dc      = math.sqrt((center_xy[0]-xt)**2 + (center_xy[1]-yt)**2)/(diam/2)
                    x.append(x_t)        
                    y.append(y_t)
                    dist2c.append( dc )
                    theta = theta + delta_a
                    cnt_nrn += 1
                    
                ring_diam = ring_diam - keepout
                if ring_diam <= 0:
                    break
            
                delta_a = (keepout*180)/(np.pi*(ring_diam/2))
            print("Placed {} neurons".format(cnt_nrn))

        elif type == "smallworld":
            import time

            max_execution_time = 0.4                    # seconds for new cluster center creation

            n = range(nb_nrn)                          
            ff = 0.997                                  # definisce numero di neuroni nel cluster -> numero di cluster [0.996-0.999] [0.997] (cambiare con attenzione)
                                                        # definisce la funzione con cui diminuisce la probabilità della posizione del nuovo neurone all'aumentare dei neuroni nel cluster
            y_th_value = 0.5                            # definisce numero di neuroni nel cluster -> numero di cluster [0.5] (cambiare con attenzione)
                                                        # definisce il valore oltre il quale deve essere il valore della funzione affinche il neurone vada nel cluster
            idx = [i for i in n if ff**i > y_th_value]  # trova il numero di neuroni che rendono la funzione maggiore di y_th_value
            nbigcluster_in_diam_org = 5                                         # definisce in modo marginale il numero di cluster
                                                                                # numero massimo di cluster adiacenti grandi che entrano diametralmente nel network
            r_max_cluster = diam/nbigcluster_in_diam_org/2                      # raggio massimo dei cluster
            rmax_centroids_center = r_max_cluster*(nbigcluster_in_diam_org-1)   # distanza massima del centro dei centroidi dal centro del network
            nsmallcluster_in_diam_org = 20                                      # definisce in modo marginale il numero di cluster
                                                                                # numero massimo di cluster adiacenti grandi che entrano diametralmente nel network
            r_min_cluster = diam/nsmallcluster_in_diam_org/2                    # raggio minimo di un cluster
            nn_max = idx[-1]                            # prende il massimo numero di neuroni che rendono la funzione maggiore di y_th_value
            ndensity = nn_max/(np.pi*r_max_cluster**2)  # densità dei neuroni nei cluster

            time_spent_creating_network = max_execution_time + 1    # inizializzazione tempo speso per la creazione network
            p_constrain = []                                        # inizializzazione vincolo pcentroid (l'inizializzazione sul tempo basta per entrare nel ciclo while)
            while not (time_spent_creating_network < max_execution_time and len(p_constrain)==0):
                start_network_creation_time = time.time()

                x = []
                y = []
                xcentroid = []
                ycentroid = []
                rcentroid = []
                pcentroid = []
                dist2c = []

                f = 1
                rand_neur_prob = 0.05   # probabilità che il primo neurone (con f = 1) non crei un cluster (per avere i neuroni random in mezzo ai cluster)
                div = 4                 # definisce numero di neuroni nel cluster -> numero di cluster [4] (cambiare con attenzione)
                                        # definisce l'intervallo entro cui verrà scelto il valore per attenuare il valore di f 
                for _ in range(nb_nrn):
                    alfa = 2 * math.pi * np.random.rand()
                    u = np.random.rand() + np.random.rand()
                    if len(xcentroid)==0:
                        r = rmax_centroids_center * (2-u if u > 1 else u)
                        xt = r * math.cos(alfa) + center_xy[0]
                        yt = r * math.sin(alfa) + center_xy[1]
                        xcentroid.append(xt)
                        ycentroid.append(yt)

                        r_temp_cluster = (r_max_cluster-r_min_cluster)*np.random.rand()+r_min_cluster # raggio cluster attuale
                        rcentroid.append(r_temp_cluster)
                        nn_temp_cluster = ndensity*np.pi*r_temp_cluster**2 # numero massimo di neuroni nel cluster con il raggio attuale 
                        ff_temp = ff**(nn_max/nn_temp_cluster) # fattore che definisce la funzione con cui diminuisce la probabilità di avere il neurone nello stesso cluster
                        f = f * ff_temp
                    else:
                        cluster_prob = np.random.rand()
                        if f*cluster_prob <= rand_neur_prob:                                                     # create a new single neuron
                            r = rmax_centroids_center * (2-u if u > 1 else u)
                            xt = r * math.cos(alfa) + center_xy[0]
                            yt = r * math.sin(alfa) + center_xy[1]
                        elif f*cluster_prob > rand_neur_prob and f*((cluster_prob+(div-1))/div) > y_th_value:    # aggiungi punto a cluster
                            r = rcentroid[-1] * (2-u if u > 1 else u)
                            xt = r * math.cos(alfa) + xcentroid[-1]
                            yt = r * math.sin(alfa) + ycentroid[-1]
                            f = f * ff_temp
                        else:                                                                                    # crea nuovo cluster  
                            pcentroid.append(f)     

                            r = rmax_centroids_center * (2-u if u > 1 else u)
                            xt = r * math.cos(alfa) + center_xy[0]
                            yt = r * math.sin(alfa) + center_xy[1]
                            r_temp_cluster = (r_max_cluster-r_min_cluster)*np.random.rand()+r_min_cluster # raggio del cluster da creare
                            nn_temp_cluster = ndensity*np.pi*r_temp_cluster**2 # numero massimo di neuroni nel cluster con il raggio attuale 
                            ff_temp = ff**(nn_max/nn_temp_cluster) # fattore che definisce la funzione con cui diminuisce la probabilità di avere il neurone nello stesso cluster
                            dd = [math.sqrt((x-xt)**2 + (y-yt)**2) for (x,y,z) in zip(xcentroid, ycentroid, rcentroid) if math.sqrt((x-xt)**2 + (y-yt)**2)<5*(r_temp_cluster+z)/4]
                            time_spent_creating_network = time.time()-start_network_creation_time
                            while not len(dd)==0 and time_spent_creating_network < max_execution_time:                                
                                alfa = 2 * math.pi * np.random.rand()
                                u = np.random.rand() + np.random.rand()
                                r = rmax_centroids_center * (2-u if u > 1 else u)
                                xt = r * math.cos(alfa) + center_xy[0]
                                yt = r * math.sin(alfa) + center_xy[1]
                                dd = [math.sqrt((x-xt)**2 + (y-yt)**2) for (x,y,z) in zip(xcentroid, ycentroid, rcentroid) if math.sqrt((x-xt)**2 + (y-yt)**2)<5*(r_temp_cluster+z)/4]
                                time_spent_creating_network = time.time()-start_network_creation_time

                            xcentroid.append(xt)
                            ycentroid.append(yt)
                            rcentroid.append(r_temp_cluster)
                            f = ff_temp

                    dc = math.sqrt((xcentroid[-1]-xt)**2 + (ycentroid[-1]-yt)**2)/r_temp_cluster
                    x.append(xt)
                    y.append(yt)
                    dist2c.append(dc)

                pcentroid.append(f)

                time_spent_creating_network = time.time() - start_network_creation_time
                p_constrain = [i for i in pcentroid if i>(np.mean(pcentroid)+2*np.std(pcentroid))]

        return [x, y, dist2c]

    def __genNrnTypes(self, nb_nrn, inh_ratio, x=[], y=[]):
        """Generate neuron types
        
        :param int nb_nrn: Number of neuron types to generate
        :param float inh_ratio: Ratio of inhibitory neurons
        :param list x: List of x coordinates of the neurons
        :param list y: List of y coordinates of the neurons

        For now independently from their position in the organoid.
        """
        tnrn = []
        for i in range (nb_nrn):
            if np.random.rand() < inh_ratio:
                tnrn.append(NRN_INH)
            else:
                tnrn.append(NRN_EXC)
        return tnrn

    def __genSynConFromFile(self, fpath):
        tsyn_row    = []
        tsyn        = []

        with open(fpath, "r") as fload:
            lines = fload.read().splitlines()
            for line in lines:
                tsyn.append(line.split(','))
        return tsyn

    # def __ruleSynConSingle(self, tsyn, ndest, nsrc, tnrn_all, x_all, y_all, org_diam, pcon):
    #     """Generate synaptic connection inside organoid based on distance between neurons
    #     :param tsyn: Array of synaptic types for all organoids
    #     :param ndest: List of destination neuron index
    #     :param nsrc: List of source neuron index
    #     :param x: Array of neuron coordinates x
    #     :param y: Array of neuron coordinates y
    #     :param org_diam: Organoid diameter (µm)
    #     :param pcon: Maximum connection probability
        
    #     Generate synaptic connection inside organoid based on distance between neurons.
    #     Neurons on the outside of the orgnaoid (higher relative distance to center) have higher connection probabilities
    #     Here, ratio is proportional to relative distance between neurons compared to the organoid diameter
    #     """

    #     cnt = 0
    #     for dest in ndest:
    #         for src in nsrc:
    #             if dest != src:
    #                 # Calculate distance between neurons
    #                 d = math.sqrt( (x_all[src]-x_all[dest])**2 + (y_all[src]-y_all[dest])**2 )
                    
    #                 # Normalize probability (the closer, the higher)
    #                 p = pcon*(1 - d/org_diam)
        
    #                 if (np.random.rand() < p):
    #                     cnt += 1
    #                     if tnrn_all[src] == NRN_INH:
    #                         tsyn[dest][src] = SYN_INH
    #                     elif tnrn_all[src] == NRN_EXC:
    #                         tsyn[dest][src] = SYN_EXC
    #     return tsyn

    # def __ruleSynConAssembloid(self, tsyn, ndest, nsrc, tnrn_all, x_all, y_all, max_d, pcon):
    #     """Generate synaptic connection for assembloid (prioritize connection to close by neurons)
    #     :param tsyn: Array of synaptic types for all organoids
    #     :param ndest: List of destination neuron index
    #     :param nsrc: List of source neuron index
    #     :param x: Array of neuron coordinates x
    #     :param y: Array of neuron coordinates y
    #     :param max_d: Maximum distance between neurons (µm)
    #     :param pcon: Maximum connection probability
        
    #     Generate synaptic connection for assembloid based on the distance between neurons.
    #     The higher the distance, the lower the connection probability
    #     Here, ratio is proportional to relative distance of the neuron compared ot max_d
    #     """

    #     cnt = 0
    #     for dest in ndest:
    #         for src in nsrc:
    #             if dest != src:
    #                 # Calculate distance between neurons
    #                 d = math.sqrt( (x_all[src]-x_all[dest])**2 + (y_all[src]-y_all[dest])**2 )
                    
    #                 # Normalize probability (the closer, the higher)
    #                 p = pcon*(1 - d/max_d)
        
    #                 if (np.random.rand() < p):
    #                     cnt += 1
    #                     if tnrn_all[src] == NRN_INH:
    #                         tsyn[dest][src] = SYN_INH
    #                     elif tnrn_all[src] == NRN_EXC:
    #                         tsyn[dest][src] = SYN_EXC
    #     return tsyn

    # def __ruleSynConConnectoid(self, tsyn, ndest, nsrc, tnrn_all, x_all, y_all, dist2c_all, pcon):
    #     """Generate synaptic connection inside organoid based on distance between neurons
    #     :param tsyn: Array of synaptic types for all organoids
    #     :param ndest: List of destination neuron index
    #     :param nsrc: List of source neuron index
    #     :param x: Array of neuron coordinates x
    #     :param y: Array of neuron coordinates y
    #     :param org_diam: Organoid diameter (µm)
    #     :param pcon: Maximum connection probability
        
    #     Generate synaptic connection inside organoid based on distance between neurons.
    #     Neurons on the outside of the orgnaoid (higher relative distance to center) have higher connection probabilities
    #     Here, ratio is proportional to relative distance between neurons compared to the organoid diameter
    #     """

    #     for dest in ndest:
    #         for src in nsrc:
    #             if dest != src:
    #                 # Normalize probability (the further from center, i.e. on the edge, the higher connectivity)
    #                 # p = pcon*(math.exp((dist2c_all[src] + dist2c_all[dest])/2)/math.exp(1)) # exponential
    #                 p = pcon*((dist2c_all[src] + dist2c_all[dest])/2) # linear
        
    #                 if (np.random.rand() < p):
    #                     if tnrn_all[src] == NRN_INH:
    #                         tsyn[dest][src] = SYN_INH
    #                     elif tnrn_all[src] == NRN_EXC:
    #                         tsyn[dest][src] = SYN_EXC
    #     return tsyn
    
    def __ruleSynConSmallWorld(self, tsyn, ndest, nsrc, tnrn_all, x_all, y_all, org_diam, pcon, config_param = pd.DataFrame()):

        for dest in ndest:
            for src in nsrc:
                if dest != src:
                    # Calculate distance between neurons
                    d = math.sqrt( (x_all[src]-x_all[dest])**2 + (y_all[src]-y_all[dest])**2 )
                    
                    # Normalize probability (the closer, the higher)
                    p = pcon*(1 - (d**(13/10))/org_diam) # <EDIT> funzione con cui varia probabilita'

                    '''provare ad aggiugere anche una probabilita che diminuisce all'aumentare del numero di connessioni che un nodo ha, 
                    quindi oltre all'effetto dato dalla distanza sulla probabilita un effetto anche dato dal numero di connesioni 
                    '''
        
                    if (np.random.rand() < p): # <EDITABLE> Qui potrei decidere la distribuzione del grado dei nodi
                        if tnrn_all[src] == NRN_INH:    # Qui si dovrebbe anche scegliere con una probabilitù se il neurone è eccitatorio o inibitorio, 
                                                        # ma la prob I/E è stata usata per definire anche il tioo di neuroni, quindi aui si usa il ragionamento che se il neurone
                                                        #  è di un certo tipo, allora la sinapsiè inibitoria , e viceversa --> aggiornamento: in letteratura RS=exc, FS=inh, quindi va bene
                            tsyn[dest][src] = SYN_INH
                        elif tnrn_all[src] == NRN_EXC:
                            if config_param.empty:
                                inh_perc = 0.5
                            else:
                                idx_slash = config_param['EXC SYNs RATIO [%]'].to_numpy()[0].index('/')
                                inh_perc = float(config_param['EXC SYNs RATIO [%]'].to_numpy()[0][idx_slash+1:])/100
                            # print(inh_perc)
                            if (np.random.rand() < inh_perc): # <EDIT>
                                tsyn[dest][src] = SYN_EXC_NMDA
                            else:
                                tsyn[dest][src] = SYN_EXC_AMPA
        return tsyn

    def __genGroupSynWeights(self, ndest, nsrc, wsyn, w):
        """Apply synaptic weight to a group"""
        for dest in ndest:
            for src in nsrc:
                wsyn[dest][src] = w
        return wsyn

    # Plotters -------------------------------------------------------------------------

    def plot(self, type, org_id=-1, org_src=-1, org_dest=-1, block=False, savepath = './'):
        """Plot parameters

        :param str type: Plot type "xy_pos", "syn_con"
        :param int org_id: Organoid id to plot for xy position
        :param int org_src: Organoid source for synaptic connection
        :param int org_dest: Organoid source for synaptic connection
        :param bool block: Set plot blocking behavior
        """
        # XY coordinates
        if   type == "xy_pos":
            if org_id != -1:
                self.__plotXYcoordinates(self.x[org_id], self.y[org_id], self.tnrn[org_id], 
                                        "Neurons organoid: {}".format(org_id), savepath = savepath)
            else:
                self.__plotXYcoordinates(self.x_all, self.y_all, self.tnrn_all, 
                                        "All neurons", savepath = savepath)
        
        # Synaptic connections
        elif type == "syn_con":
            if (org_src != -1) and (org_dest != -1):
                self.__plotSynCon(self.nlist[org_dest], self.nlist[org_src], self.tsyn, self.x_all, self.y_all, self.tnrn_all,  
                                "Synaptic connection organoid {} to organoid {}".format(org_src, org_dest), savepath = savepath)
            else:
                self.__plotSynCon(self.nlist_all, self.nlist_all, self.tsyn, self.x_all, self.y_all, self.tnrn_all,
                                "All synaptic connections", savepath = savepath)
        
        plt.show(block=block)

    def __plotXYcoordinates(self, x, y, tnrn=[], title="", savepath = './'):
        """Plot XY coordinates
        
        :param list x: List of x coordinates of neurons
        :param list y: List of y coordinates of neurons
        :param list tnt: List of neurons' types
        :param str title: Figure title
        """
        fig = plt.figure("Print XY coordinates: " + title)
        fig.suptitle(title, fontsize=20)
        plt.xlabel('X (um)', fontsize=18)
        plt.ylabel('Y (um)', fontsize=16)
        
        if tnrn:
            colors = [COLOR_INH if t==NRN_INH else COLOR_EXC for t in tnrn]
            plt.scatter(x, y, label=",", color=colors,  marker=".", s=10)
        else:
            plt.scatter(x, y, label=",", color="black", marker=".", s=10)

        for t in ['svg','jpg']:
            plt.savefig(savepath + "." + t, format = t)
        plt.show(block=False)
    
    def __plotSynCon(self, ndest, nsrc, tsyn, x_all, y_all, tnrn_all, title="", savepath = './'):
        """Plot synaptic connections"""
        fig = plt.figure("Print XY coordinates: " + title)
        fig.suptitle(title, fontsize=20)
        plt.xlabel('X (um)', fontsize=18)
        plt.ylabel('Y (um)', fontsize=16)

        # Neurons
        colors = [COLOR_INH if t==NRN_INH else COLOR_EXC for t in tnrn_all]
        plt.scatter(x_all, y_all, label= ",", color=colors, marker= ".", s=10)

        # Connections
        cnt = 0
        for dest in ndest:
            for src in nsrc:
                if tsyn[dest][src] != SYN_NONE:                    
                    if   tnrn_all[src] == NRN_INH: linecolor = COLOR_INH
                    elif tnrn_all[src] == NRN_EXC: linecolor = COLOR_EXC
                    else:                          linecolor = "black"
                    plt.plot([x_all[src], x_all[dest]], [y_all[src], y_all[dest]], color=linecolor, linewidth=0.1)
        
        for t in ['svg','png']:
            plt.savefig(savepath + "." + t, format = t)
        plt.show(block=False)

    # Getters -------------------------------------------------------------------------

    def getSynTypes(self):
        """Get types of synapses in SNN-HH encoding"""
        tsyn_row_str = []
        tsyn_str = []
        for dest in range(self.nb_nrn):
            for src in range(self.nb_nrn):
                tsyn_row_str.append(syncode[self.tsyn[dest][src]])
            tsyn_str.append(tsyn_row_str)
            tsyn_row_str = []
        return tsyn_str

    def getSynWeights(self):
        """Get weight of synapses"""
        return self.wsyn

    def getNeuronTypes(self):
        """Get neuron types in SNN-HH encoding"""
        return [nrncode[n] for n in self.tnrn_all]

    def getPosXY(self, org_id=-1):
        """Get xy position of the neurons of an organoid or all organoids"""
        if org_id == -1:
            return [self.x_all, self.y_all]
        else:
            return [self.x[org_id], self.y[org_id]]