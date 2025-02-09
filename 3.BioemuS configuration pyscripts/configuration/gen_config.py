# -*- coding: utf-8 -*-
# @title      Generate configuration files for SNN HH
# @file       gen_config.py
# @author     Romain Beaubois
# @date       23 Oct 2023
# @copyright
# SPDX-FileCopyrightText: © 2023 Romain Beaubois <refbeaubois@yahoo.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# @brief Script to generate configuration file for SNN HH
# 
# @details 
# > **23 Oct 2023** : file creation (RB)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# np.random.seed(333)

from configuration.file_managers.HwConfigFile     import *
from configuration.file_managers.SwConfigFile     import *
from configuration.neurons.Ionrates               import *
from configuration.neurons.Hhparam                import *
from configuration.synapses.Synapses              import *
from configuration.network_models.OrgStructures   import *
from configuration.network_models.RatBrain        import *
from configuration.utility.settings               import *

def gen_config(config_name:str, save_path:str="./", en_stim=False, stim_delay_ms=0, stim_duration_ms=0, config_param = pd.DataFrame()):
    # System parameters ####################################################################
    # Hardware platform (from KR260 platform)
    sw_ver              = SOFTWARE_VERSION
    NB_NEURONS          = HW_MAX_NB_NEURONS
    dt                  = HW_DT

    # Files
    config_fname        = config_name
    local_dirpath_save  = save_path

    # Model selection
    MODEL                       = "ratbrain"   # Model to generate: "organoid" "custom" "ratbrain"

    # FPGA dev
    GEN_SIM_DEBUG_DATA          = False

    # Application parameters ################################################################
    swconfig_builder                                           = SwConfigFile()
    swconfig_builder.parameters["fpath_hwconfig"]              = "/home/ubuntu/bioemus/config/hwconfig_" + config_fname + ".txt"
    swconfig_builder.parameters["emulation_time_s"]            = 600                                    # <EDIT>
    swconfig_builder.parameters["sel_nrn_vmem_dac"]            = [n for n in range(8)]
    swconfig_builder.parameters["sel_nrn_vmem_dma"]            = [n for n in range(16)]
    swconfig_builder.parameters["save_local_spikes"]           = True
    swconfig_builder.parameters["save_local_vmem"]             = False
    swconfig_builder.parameters["save_path"]                   = "/home/ubuntu/bioemus/data/" # target saving director
    swconfig_builder.parameters["en_zmq_spikes"]               = False
    swconfig_builder.parameters["en_zmq_vmem"]                 = False
    swconfig_builder.parameters["en_zmq_stim"]                 = False
    swconfig_builder.parameters["en_wifi_spikes"]              = False
    swconfig_builder.parameters["ip_zmq_spikes"]               = "tcp://*:5557"
    swconfig_builder.parameters["ip_zmq_vmem"]                 = "tcp://*:5558"
    swconfig_builder.parameters["ip_zmq_stim"]                 = "tcp://192.168.137.1:5559"
    swconfig_builder.parameters["nb_tstamp_per_spk_transfer"]  = 100
    swconfig_builder.parameters["nb_tstep_per_vmem_transfer"]  = 190
    swconfig_builder.parameters["en_stim"]                     = False
    swconfig_builder.parameters["stim_delay_ms"]               = stim_delay_ms
    swconfig_builder.parameters["stim_duration_ms"]            = stim_duration_ms

    # Globals & Builders ####################################################################
    tsyn_row, wsyn_row    = ([] for i in range(2))
    tsyn,     wsyn        = ([] for i in range(2))
    tnrn                  = []

    #   ██████ ██    ██ ███████ ████████  ██████  ███    ███ 
    #  ██      ██    ██ ██         ██    ██    ██ ████  ████ 
    #  ██      ██    ██ ███████    ██    ██    ██ ██ ████ ██ 
    #  ██      ██    ██      ██    ██    ██    ██ ██  ██  ██ 
    #   ██████  ██████  ███████    ██     ██████  ██      ██ 
    #                                                        
    # Custom model #################################################################
    if MODEL == "custom":
        tnrn    = ["FS_nonoise"]*NB_NEURONS

        # tnrn[0] = "FS"
        # for i in range(0,511,1):
        #     tnrn[i]  = "FSorg"
        # for i in range(512,1023,1):
        #     tnrn[i]  = "RSorg"

        tnrn[0] = "FS_nonoise"
        tnrn[1] = "RS_nonoise"
        tnrn[2] = "IB_nonoise"
        tnrn[3] = "LTS_nonoise"

        SYN_MODE = "NONE"
        # SYN_MODE = "CHASER"
        # SYN_MODE = "RANDOM"
        # SYN_MODE = "ONE_TO_ALL"
        # SYN_MODE = "ONE_TO_ONE"

        # Create synaptic conncetions

        # Synaptic types
        #      | source |
        # -----|--------|
        # dest |        |
        tsyn_dict = Synapses().getDict()
        weight = 1.9
        for dest in range(NB_NEURONS):
            for src in range(NB_NEURONS):

                if SYN_MODE == "NONE":
                    tsyn_i = "destexhe_none"

                elif SYN_MODE == "CHASER":
                    if ((src+1) == dest):
                        tsyn_i = "destexhe_ampa"
                    else:
                        tsyn_i = "destexhe_none"

                elif SYN_MODE == "RANDOM":
                    if (dest < 100) and (src < 100):
                        if dest != src:
                            if (np.random.rand() < 0.2):
                                tsyn_i = "destexhe_ampa"
                            else:
                                tsyn_i = "destexhe_none"
                        else:
                            tsyn_i = "destexhe_none"
                    else:
                        tsyn_i = "destexhe_none"

                elif SYN_MODE == "ONE_TO_ONE":
                    if src==0 and dest==1:
                        tsyn_i = "destexhe_gabab"
                    elif src==1 and dest==2:
                        tsyn_i = "destexhe_gabab"
                    else:
                        tsyn_i = "destexhe_none"

                elif SYN_MODE == "ONE_TO_ALL":
                    if src==0 and dest != 0:
                        tsyn_i = "destexhe_gabaa"
                    else:
                        tsyn_i = "destexhe_none"

                tsyn_row.append(tsyn_dict[tsyn_i])
                if tsyn_i == "destexhe_none":
                    wsyn_row.append(0.0)
                else:
                    if tsyn_i == "destexhe_ampa" or tsyn_i == "destexhe_gabaa":
                        wsyn_row.append(1.0)
                    else:
                        wsyn_row.append(weight)

            tsyn.append(tsyn_row)
            wsyn.append(wsyn_row)
            tsyn_row = []
            wsyn_row = []

    #   ██████  ██████   ██████   █████  ███    ██  ██████  ██ ██████  
    #  ██    ██ ██   ██ ██       ██   ██ ████   ██ ██    ██ ██ ██   ██ 
    #  ██    ██ ██████  ██   ███ ███████ ██ ██  ██ ██    ██ ██ ██   ██ 
    #  ██    ██ ██   ██ ██    ██ ██   ██ ██  ██ ██ ██    ██ ██ ██   ██ 
    #   ██████  ██   ██  ██████  ██   ██ ██   ████  ██████  ██ ██████  
    #                                                                  
    # Organoid modeling #################################################################
    elif MODEL == "organoid":
        # Instanciate helper for organoid modeling configuration
        org         = OrgStructures(NB_NEURONS)

        # Configure organoid model
        ## (1) - Add organoids
        org.addOrganoid(org_diam=250, nrn_diam=15, org_center_xy=[0, 0])  
        org.addOrganoid(org_diam=250, nrn_diam=15, org_center_xy=[500, 0])

        ## (2) - Generate neurons
        org.genNeurons(inh_ratio=0.2)

        ## (3) - Generate synaptic connections
        org.genSynCon(rule="single",     org_src=0, org_dest=0, max_pcon=0.10)
        org.genSynCon(rule="single",     org_src=1, org_dest=1, max_pcon=0.10)
        org.genSynCon(rule="assembloid", org_src=0, org_dest=1, max_pcon=0.03)
        org.genSynCon(rule="assembloid", org_src=1, org_dest=0, max_pcon=0.03)

        ## (4) - Assign weights
        org.genSynWeights(org_src= 0, org_dest=0, weight=1.0)
        org.genSynWeights(org_src= 1, org_dest=1, weight=1.0)
        org.genSynWeights(org_src= 0, org_dest=1, weight=1.0)
        org.genSynWeights(org_src= 1, org_dest=0, weight=1.0)

        # Print
        # org.plot("xy_pos")
        # org.plot("syn_con", org_src=0, org_dest=0)
        # org.plot("syn_con", org_src=1, org_dest=1)
        # org.plot("syn_con", org_src=0, org_dest=1)
        # org.plot("syn_con", org_src=1, org_dest=0)
        # org.plot("syn_con", block=True)

        # --------------------------------
        # NO NEED TO EDIT UNDER
        # (format for hardware config)

        # Get model parameters
        tsyn_org    = org.getSynTypes()
        wsyn_org    = org.getSynWeights()
        tnrn_org    = org.getNeuronTypes()
        tsyn_dict   = Synapses().getDict()

        for dest in range(NB_NEURONS):
            for src in range(NB_NEURONS):
                tsyn_i = tsyn_org[dest][src]
                wsyn_i = wsyn_org[dest][src]
                tsyn_row.append(tsyn_dict[tsyn_i])
                if tsyn_i == "destexhe_none":
                    wsyn_row.append(0.0)
                elif tsyn_i == "destexhe_ampa":
                    wsyn_row.append(0.10*wsyn_i)
                elif tsyn_i == "destexhe_gabaa":
                    wsyn_row.append(wsyn_i)
                else:
                    wsyn_row.append(wsyn_i)

            tsyn.append(tsyn_row)
            wsyn.append(wsyn_row)

            tsyn_row = []
            wsyn_row = []
        
        tnrn = tnrn_org
    # RFA modeling #################################################################
    elif MODEL == "ratbrain":
        # Instanciate helper for organoid modeling configuration
        org         = RatBrain(NB_NEURONS)

        # Configure organoid model
        ## (1) - Add organoids
        NB_ORGANOID = 1                                                             # <EDIT>
        for i in range(NB_ORGANOID):
            org.addOrganoid(org_diam=250, nrn_diam=15, org_center_xy=[500*i, 0])    # <EDIT>

        ## (2) - Generate neurons
        org.genNeurons(inh_ratio=0.215)                                               # <EDIT>

        ## (3) - Generate synaptic connections
        for src in range(NB_ORGANOID):
            for dest in range(NB_ORGANOID):
                if src == dest: # within networks
                    org.genSynCon(rule="smallworld",     org_src=src, org_dest=dest, max_pcon=0.3, config_param=config_param) # <EDIT>
                else: # between networks
                    org.genSynCon(rule="smallworld",     org_src=src, org_dest=dest, max_pcon=0.03) # <EDIT>

        ## (4) - Assign weights
        for src in range(NB_ORGANOID):
            for dest in range(NB_ORGANOID):
                if src == dest: # within networks
                    org.genSynWeights(org_src= src, org_dest=dest, weight=1.0) # <EDIT> tutte le sinapsi sono uguali
                else: # between networks
                    org.genSynWeights(org_src= src, org_dest=dest, weight=1.0) # <EDIT> tutte le sinapsi sono uguali

        # Print
        org.plot("xy_pos", savepath = os.path.join(local_dirpath_save, "xyPos_" + config_fname))
        # org.plot("syn_con", org_src=0, org_dest=0)
        # org.plot("syn_con", org_src=1, org_dest=1)
        # org.plot("syn_con", org_src=0, org_dest=1)
        # org.plot("syn_con", org_src=1, org_dest=0)
        org.plot("syn_con", block=True, savepath = os.path.join(local_dirpath_save, "synCon_" + config_fname))

        # --------------------------------
        # NO NEED TO EDIT UNDER
        # (format for hardware config)

        # Get model parameters
        tsyn_org    = org.getSynTypes()
        wsyn_org    = org.getSynWeights()
        tnrn_org    = org.getNeuronTypes()
        tsyn_dict   = Synapses().getDict()

        for dest in range(NB_NEURONS):
            for src in range(NB_NEURONS):
                tsyn_i = tsyn_org[dest][src]
                wsyn_i = wsyn_org[dest][src]
                tsyn_row.append(tsyn_dict[tsyn_i])
                if tsyn_i == "destexhe_none":
                    wsyn_row.append(0.0)
                elif tsyn_i == "destexhe_ampa":
                    wsyn_row.append(0.10*wsyn_i) # <EDITABLE>
                elif tsyn_i == "destexhe_gabaa":
                    wsyn_row.append(wsyn_i)
                elif tsyn_i == "destexhe_nmda":
                    if config_param.empty:
                        wnmda=0.4
                    else:
                        wnmda = float(config_param['WSYN NMDA'].to_numpy()[0])
                    # print(wnmda)
                    wsyn_row.append(wnmda*wsyn_i) # <EDITABLE>
                elif tsyn_i == "destexhe_gabab":
                    wsyn_row.append(wsyn_i)
                else:
                    wsyn_row.append(wsyn_i)

            tsyn.append(tsyn_row)
            wsyn.append(wsyn_row)

            tsyn_row = []
            wsyn_row = []
        
        tnrn = tnrn_org

        xy_pos = org.getPosXY()
    else:
        exit()

    # Write file
    with open(os.path.join(local_dirpath_save, "xyPos_" + config_fname + ".txt"), 'w') as f_xyPos:
        json.dump(xy_pos, f_xyPos)
    with open(os.path.join(local_dirpath_save, "nType_" + config_fname + ".txt"), 'w') as f_nType:
        json.dump(tnrn_org, f_nType)
    with open(os.path.join(local_dirpath_save, "sType_" + config_fname + ".txt"), 'w') as f_sType:
        json.dump(tsyn_org, f_sType)



    #   ██████  ██████  ███    ██ ███████ ██  ██████      ███████ ██ ██      ███████ 
    #  ██      ██    ██ ████   ██ ██      ██ ██           ██      ██ ██      ██      
    #  ██      ██    ██ ██ ██  ██ █████   ██ ██   ███     █████   ██ ██      █████   
    #  ██      ██    ██ ██  ██ ██ ██      ██ ██    ██     ██      ██ ██      ██      
    #   ██████  ██████  ██   ████ ██      ██  ██████      ██      ██ ███████ ███████ 
    #                                                                                
    # Config file #################################################################
    hw_cfg_file                 = HwConfigFile(sw_ver, NB_NEURONS)

    # Parameters
    hw_cfg_file.dt              = dt
    hw_cfg_file.nb_hhparam      = Hhparam().getNb()
    hw_cfg_file.nb_ionrate      = Ionrates().getNbIonRates("pospischil")
    hw_cfg_file.depth_ionrate   = Ionrates().getDepthIonRates("pospischil")
    hw_cfg_file.depth_synrate   = Synapses().getDepthSynRates("destexhe")

    # Ionrates
    [hw_cfg_file.m_rates1, hw_cfg_file.m_rates2,
    hw_cfg_file.h_rates1, hw_cfg_file.h_rates2] = Ionrates().getIonRates("pospischil", dt, GEN_SIM_DEBUG_DATA)

    # Synapse parameters
    hw_cfg_file.psyn     = Synapses().getPsyn("destexhe", dt)

    # Synrates
    hw_cfg_file.synrates = Synapses().getSynRates("destexhe", GEN_SIM_DEBUG_DATA)

    # Neuron types
    for n in tnrn:
        hhp = Hhparam().getParameters(n, dt, config_param=config_param)

        # Randomize noise parameters
        # [hhp, _, _] = Hhparam().getParameters(n, dt)
        # dp = nhh.getDictHHparam()
        # hhp[dp["mu"]]       = hhp[dp["mu"]]    + 0.02*(np.random.rand()*hhp[dp["mu"]]      - np.random.rand()*hhp[dp["mu"]])
        # hhp[dp["theta"]]    = hhp[dp["theta"]] + 0.02*(np.random.rand()*hhp[dp["theta"]]   - np.random.rand()*hhp[dp["theta"]])
        # hhp[dp["sigma"]]    = hhp[dp["sigma"]] + 0.02*(np.random.rand()*hhp[dp["sigma"]]   - np.random.rand()*hhp[dp["sigma"]])

        hw_cfg_file.HH_param.append(hhp)

    # Synapses
    hw_cfg_file.tsyn = tsyn
    hw_cfg_file.wsyn = wsyn

    # Write file
    swconfig_builder.write(os.path.join(local_dirpath_save, "swconfig_" + config_fname + ".json"))  # save path of swconfig on local
    hw_cfg_file.write(os.path.join(local_dirpath_save, "hwconfig_" + config_fname + ".txt"))        # save path of hwconfig on local

    return [hw_cfg_file, swconfig_builder]