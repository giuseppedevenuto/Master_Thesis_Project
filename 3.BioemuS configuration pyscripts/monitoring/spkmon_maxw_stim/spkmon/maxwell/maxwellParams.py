# -*- coding: utf-8 -*-
# @title      Parameters defines for maxwell
# @file       maxwellParams.py
# @author     Romain Beaubois
# @date       24 Aug 2022
# @copyright
# SPDX-FileCopyrightText: © 2022 Romain Beaubois <refbeaubois@yahoo.com>
# SPDX-License-Identifier: MIT
#
# @brief
# 
# @details
# > **24 Aug 2022** : file creation (RB)


# Maxwell system default settings
MAX_NB_EL           = 26400     # Maximum number of electrodes
MAX_NB_RECORDING_EL = 1024      # Maximum number of recording electrodes
MAX_NB_STIM_EL      = 32        # Maximum number of stimulation electrodes
FS                  = 20e3      # Sampling frequency (Hz)
NB_EL_PER_LINE      = 220

# Messages types
MSG_TYPE = {    "INFO"      : "\N{BLACK RIGHTWARDS ARROWHEAD} ",
                "SUCCESS"   : "\N{LARGE GREEN CIRCLE} ",
                "WARNING"   : "\N{LARGE ORANGE CIRCLE} ",
                "FAILURE"   : "\N{LARGE RED CIRCLE} ",
                "DEBUG"     : "\N{BLACK FLAG} ",
                "ERROR"     : "\N{WARNING SIGN} "
            }