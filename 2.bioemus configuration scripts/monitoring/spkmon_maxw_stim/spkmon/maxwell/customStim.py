# -*- coding: utf-8 -*-
# @title      Template for custom stimulation experiments
# @file       templateCustomStimExp.py
# @author     Romain Beaubois
# @date       22 Aug 2022
# @copyright
# SPDX-FileCopyrightText: © 2022 Romain Beaubois <refbeaubois@yahoo.com>
# SPDX-License-Identifier: MIT
#
# @brief
# 
# @details
# > **22 Aug 2022** : file creation (RB)

######################################################################
# Imports
######################################################################
from datetime import datetime
import maxlab
import maxlab.system
import maxlab.chip
import maxlab.util
import maxlab.saving
import time
from spkmon.maxwell.seqGenerator import *
from spkmon.maxwell.maxwellParams import *

######################################################################
# USER
######################################################################
# Recording settings
DATE                = '220822'
CULTURE_ID          = 'C5'
EXP_ID              = 'Exp4'
FPATH_CFG           = 'test.cfg' # Path to the config file you want to use
DIRPATH_DATA        = './' # Path of the directory where the recording and sequence log should be stored
FNAME_DATA          = DATE + '_' + CULTURE_ID + EXP_ID # Name the recording file should have
USE_LEGACY_WRITE    = True # Data format (True for legacy format)
RECORD_ONLY_SPIKES  = False # Select record spikes only or signals as well
LIST_WELLS          = range(1) # Wells to record from (range(1) for MaxOne, range(6) for MaxTwo)

# Experiment protocol
STIM_ON_TIME_S      = 0.050 # Stimulation on time (s)

# Stimulation electrodes
LIST_STIM_EL = [4507, 4529, 9127, 8930, 6939, 6940] # Stimulation electrodes

seq = maxlab.Sequence()
stimulation_units = []

def initMaxwellStim():
    # Stimulation sequence
    print(MSG_TYPE["INFO"] + "Generate sequence")
    print(datetime.now())

    freq_hz = 100
    for nb_pattern in range(int(STIM_ON_TIME_S*freq_hz)):
        append_stimulation(seq, type="sin", freq=freq_hz, max_amp=40.0)

    print(datetime.now())

    ######################################################################
    # 0. Initialize system into a defined state
    ######################################################################

    print(MSG_TYPE["INFO"] + "Initialize system")
    maxlab.util.initialize()
    maxlab.send(maxlab.chip.Core().enable_stimulation_power(True))


    ######################################################################
    # 1. Select electrodes to record and stimulate
    ######################################################################

    print(MSG_TYPE["INFO"] + "Select electrodes to record and stimulate")
    array = maxlab.chip.Array('stimulation')
    array.reset()
    array.load_config(FPATH_CFG)


    ######################################################################
    # 2. Connect stimulation units to the stimulation electrodes
    ######################################################################

    print(MSG_TYPE["INFO"] + "Connect stimulation units to stimulation electrodes")

    for stim_el in LIST_STIM_EL:
        array.connect_electrode_to_stimulation( stim_el )
        stim = array.query_stimulation_at_electrode( stim_el )
        if stim:
            stimulation_units.append( stim )
        else:
            print(MSG_TYPE["FAILURE"] + "No stimulation channel can connect to electrode: " + str(stim_el))

    array.download()

    maxlab.util.offset()


    ######################################################################
    # 3. Power up and configure the stimulation units
    ######################################################################

    print(MSG_TYPE["INFO"] + "Setup stim units")
    stimulation_unit_commands = []

    for stimulation_unit in stimulation_units:
        # Stimulation Unit
        stim = maxlab.chip.StimulationUnit(stimulation_unit)
        stim.power_up(True)
        stim.connect(True)
        stim.set_voltage_mode()
        stim.dac_source(0)
        stimulation_unit_commands.append(stim)
        maxlab.send(stim)


    ######################################################################
    # 4. Recording and stimulation
    ######################################################################

    # First, poweroff all the stimulation units
    for stimulation_unit in range(0,32):
        # Stimulation Unit
        stim = maxlab.chip.StimulationUnit(stimulation_unit)
        stim.power_up(False)
        stim.connect(False)
        maxlab.send(stim)

    # # Start recording
    # s = maxlab.saving.Saving()
    # s.open_directory(DIRPATH_DATA)
    # s.set_legacy_format(USE_LEGACY_WRITE)
    # s.group_delete_all()
    # if not RECORD_ONLY_SPIKES:
    #     for well in LIST_WELLS:
    #         s.group_define(well, "routed")

    # s.start_file(FNAME_DATA)
    # s.start_recording(LIST_WELLS)

    # Start with stim off
    print(MSG_TYPE["DEBUG"] + "Waiting for bursts")

def sendMaxwellStim(stim_trig:bool):
    # Power on all units
    for stimulation_unit in stimulation_units:
        # Power up stimulation unit
        print(MSG_TYPE["INFO"] + "Power up stimulation unit " + str(stimulation_unit))
        stim = maxlab.chip.StimulationUnit(stimulation_unit)
        stim.power_up(True).connect(True).set_voltage_mode().dac_source(0)
        maxlab.send(stim)

    # Send stimulation sequence
    print(MSG_TYPE["DEBUG"] + "Stim ON")
    print(MSG_TYPE["INFO"] + "Send stimulation sequence")
    seq.send()

    # Power down all stimulation units
    for stimulation_unit in stimulation_units:
        print(MSG_TYPE["INFO"] + "Power down stimulation unit " + str(stimulation_unit))
        stim = maxlab.chip.StimulationUnit(stimulation_unit).power_up(False)
        maxlab.send(stim)

    # Stop recording
    # print(MSG_TYPE["INFO"] + "Stop recording")
    # s.stop_recording()
    # s.stop_file()
    # s.group_delete_all()