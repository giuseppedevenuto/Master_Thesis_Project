# -*- coding: utf-8 -*-
# @title      Sequence generator
# @file       seqGenerator.py
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

######################################################################
# Imports
######################################################################
import maxlab
import maxlab.system
import maxlab.chip
import maxlab.util

from spkmon.maxwell.maxwellParams import FS, MSG_TYPE

import math
import numpy as np
from numpy.random import randn

######################################################################
# Constants
######################################################################

MAX_STIM_AMP_MV     = 1484      # Max stimulation amplitude (mV)
MIN_STIM_AMP_MV     = -1484     # Min stimulation amplitude (mV)
NO_STIM_AMP_MV      = 0         # Ampltiude for no stimulation (mV)
MAX_FREQ_SIN        = FS/10     # considering 10 points minimum

######################################################################
# Functions
######################################################################

def mVtoDAC(val_mv:float, stim_mode="voltage", max_amp=MAX_STIM_AMP_MV, min_amp=MIN_STIM_AMP_MV):
    """mV to DAC value"""
    DAC_RESOLUTION_MV   = 2.9       # DAC resolution (mV)
    DAC_RANGE_BIT       = 1024      # Range for DAC value (10 bits DAC so 2^10)
    DAC_ZERO_BIT        = int(DAC_RANGE_BIT/2)  # DAC value for 0 mV

    # Check for stimulation amplitude applied
    if val_mv > MAX_STIM_AMP_MV:
        val_mv = MAX_STIM_AMP_MV
        print(MSG_TYPE["WARNING"] + "Stimulation max amplitude clipped to" + MAX_STIM_AMP_MV)
    if val_mv < MIN_STIM_AMP_MV:
        val_mv = MIN_STIM_AMP_MV
        print(MSG_TYPE["WARNING"] + "Stimulation min amplitude clipped to" + MIN_STIM_AMP_MV)

    # DAC in voltage mode is inverter amp so (-) to make (+)
    if stim_mode == "voltage" :
        return DAC_ZERO_BIT - round(val_mv/DAC_RESOLUTION_MV)
    else:
        return DAC_ZERO_BIT + round(val_mv/DAC_RESOLUTION_MV)

def append_stimulation(seq, type, freq, max_amp=MAX_STIM_AMP_MV, min_amp=0, 
                        duty_cycle=50, pulse_width=100,
                        theta=1.0, mu=0.1, sigma=1.5, duration=100):
    """Generate stimulation sequence

    :param maxlab.Sequence seq: Stimulation sequence
    :param str type: Stimulate type ("bi-phasic", "square", "pulse", "sin", "noise", "synaptic noise")
    :param float freq: Stimulation frequency (Hz)
    :param int max_amp: Maximum amplitude of stimulation (MIN_STIM_AMP_MV to MAX_STIM_AMP_MV)
    :param int min_amp: Minimum amplitude of stimulation (MIN_STIM_AMP_MV to MAX_STIM_AMP_MV)
    :param int duty_cycle: Duty cycle for square signals (%)
    :param int pulse_width: Pulse width for pulse signal and bi-phasic (µs)
    :param float theta: Theta for OU based noise
    :param float mu: Mu for OU based noise
    :param float sigma: Sigma for OU based noise
    :param int duration: Duration of noise pattern (µs)

    :returns: stimulation sequence
    :rtype: maxlab.Sequence
    """

    if type == "bi-phasic":
        bi_phasic_seq(seq, freq, max_amp, min_amp, pulse_width)
    elif type == "square":
        square_seq(seq, freq, max_amp, min_amp, duty_cycle)
    elif type == "pulse":
        pulse_seq(seq, freq, max_amp, min_amp, pulse_width)
    elif type == "sin":
        sin_seq(seq, freq, max_amp)
    elif type == "noise":
        noise_seq(seq, freq, max_amp)
    elif type == "synaptic noise":
        syn_noise_seq(seq, max_amp, theta, mu, sigma, duration)

def bi_phasic_seq(seq, freq, max_amp:int, min_amp:int, pulse_width):
    """Bi-phasic pulse stimulation"""

    Ts_us                   = (1e6/FS)                  # Sampling period (µs)
    pulse_width_in_sample   = int(pulse_width/Ts_us)    # Sample width in number of samples
    period_in_sample        = int((1e6/freq)/Ts_us)     # Signal period in number of samples

    # High state pulse
    seq.append( maxlab.chip.DAC(0, mVtoDAC(max_amp)) ) # DAC in voltage mode is inverter amp so (-) to make (+) 
    seq.append( maxlab.system.DelaySamples(pulse_width_in_sample) )

    # Low state pulse
    seq.append( maxlab.chip.DAC(0, mVtoDAC(min_amp)) ) # DAC in voltage mode is inverter amp so (-) to make (+) 
    seq.append( maxlab.system.DelaySamples(pulse_width_in_sample) )

    # Back to zero
    seq.append( maxlab.chip.DAC(0, mVtoDAC(0.0)) )
    seq.append( maxlab.system.DelaySamples(period_in_sample-pulse_width_in_sample*2) )

    return seq

def square_seq(seq, freq, max_amp:int, min_amp:int, duty_cycle):
    """Square wave stimulation"""

    Ts_us                   = (1e6/FS)  # Sampling period (µs)
    period_in_sample        = int((1e6/freq)/Ts_us) # Signal period in samples
    Ton_in_sample           = int(period_in_sample*(duty_cycle/100)) # Period at high state in samples
    Toff_in_sample          = int(period_in_sample*(duty_cycle/100)) # Period at low state in samples

    # High state
    seq.append( maxlab.chip.DAC(0, mVtoDAC(max_amp)) )
    seq.append( maxlab.system.DelaySamples(Ton_in_sample) )

    # Low state
    seq.append( maxlab.chip.DAC(0, mVtoDAC(min_amp)) )
    seq.append( maxlab.system.DelaySamples(Toff_in_sample) )

    return seq

def pulse_seq(seq, freq, max_amp:int, min_amp:int, pulse_width):
    """Pulse stimulation"""

    Ts_us                   = (1e6/FS)                  # Sampling period (µs)
    pulse_width_in_sample   = int(pulse_width/Ts_us)    # Pulse width in samples
    period_in_sample        = int((1e6/freq)/Ts_us)     # Signal period in samples

    if max_amp > MAX_STIM_AMP_MV:
        max_amp = MAX_STIM_AMP_MV
        print(MSG_TYPE["WARNING"] + "Stimulation max amplitude clipped to" + MAX_STIM_AMP_MV)
    if min_amp < MIN_STIM_AMP_MV:
        min_amp = MIN_STIM_AMP_MV
        print(MSG_TYPE["WARNING"] + "Stimulation min amplitude clipped to" + MIN_STIM_AMP_MV)

    # High state
    seq.append( maxlab.chip.DAC(0, mVtoDAC(max_amp)) )
    seq.append( maxlab.system.DelaySamples(pulse_width_in_sample) )

    # Low state
    seq.append( maxlab.chip.DAC(0, mVtoDAC(min_amp)) )
    seq.append( maxlab.system.DelaySamples(period_in_sample-pulse_width_in_sample) )

    return seq

def sin_seq(seq, freq, max_amp):
    """Sinusoidal wave stimulation"""

    Ts_us               = (1e6/FS) # Sampling period (µs)
    T_us                = (1e6/freq) # Signal period (µs)
    l_t_in_samples      = int(T_us/Ts_us) # Number of sample per period
    t                   = np.linspace(-Ts_us, Ts_us, l_t_in_samples) # Time points

    if max_amp > MAX_STIM_AMP_MV:
        max_amp = MIN_STIM_AMP_MV
        print(MSG_TYPE["WARNING"] + "Stimulation max amplitude clipped to" + MAX_STIM_AMP_MV)    
    
    for i in range(l_t_in_samples):
        val = max_amp * math.sin(2*math.pi*freq*t[i])
        seq.append( maxlab.chip.DAC(0, mVtoDAC(val)) )
        seq.append( maxlab.system.DelaySamples(1) )
    return seq

def noise_seq(seq, freq, max_amp):
    """Normally distributed noise stimulation"""
    Ts_us               = (1e6/FS) # Sampling period (µs)
    T_us                = (1e6/freq) # Signal period (µs)
    l_t_in_samples      = int(T_us/Ts_us) # Number of sample per period
    t                   = np.linspace(-Ts_us, Ts_us, l_t_in_samples) # Time vector

    for _ in range(l_t_in_samples):
        val = int(max_amp*(randn()/4))

        if val > MAX_STIM_AMP_MV:
            val = min(val, MAX_STIM_AMP_MV)
            print(MSG_TYPE["WARNING"] + "Stimulation max amplitude clipped to" + MAX_STIM_AMP_MV)
        if val < MIN_STIM_AMP_MV:
            val = max(val, MIN_STIM_AMP_MV)
            print(MSG_TYPE["WARNING"] + "Stimulation min amplitude clipped to" + MIN_STIM_AMP_MV)

        seq.append( maxlab.chip.DAC(0, mVtoDAC(val)) )
        seq.append( maxlab.system.DelaySamples(1) )
    return seq

def syn_noise_seq(seq, max_amp, theta, mu, sigma, duration):
    """Synaptic noise stimulation (Destexhe article based on Ornstein-Oulenbeck process)"""
    dt_ms           = 1e3/FS    # Time step (ms)
    Ts_us           = (1e6/FS)  # Sampling period (µs)
    l_t_in_samples  = int(duration/Ts_us) # Period in samples
    oprev           = mu # Initial value

    for _ in range(l_t_in_samples):
        dW      = (2e-4)*randn(1,1)
        o       = oprev + theta*dt_ms*(mu - oprev) + sigma*dW
        oprev   = o
        val     = int((max_amp/mu)*o)

        if val > MAX_STIM_AMP_MV:
            val = min(val, MAX_STIM_AMP_MV)
            print(MSG_TYPE["WARNING"] + "Stimulation max amplitude clipped to" + MAX_STIM_AMP_MV)
        if val < MIN_STIM_AMP_MV:
            val = max(val, MIN_STIM_AMP_MV)
            print(MSG_TYPE["WARNING"] + "Stimulation min amplitude clipped to" + MIN_STIM_AMP_MV)

        seq.append( maxlab.chip.DAC(0, mVtoDAC(val)) )
        seq.append( maxlab.system.DelaySamples(1) )
    return seq