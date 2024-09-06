#!/usr/bin/env python3

"""Mapping functions and preprogrammed outputs for haptic display.

There are 2 classes defined in this script:
- IntensityMap: Functions to convert a 0-1 intensity into haptic output frequency, duty cycle, amplitude
- IntensityGenerator: Functions to create pre-programmed intensity sequences for the haptic display."""

import numpy as np
from scipy import signal

############### INTENSITY MAPPINGS + CLASS ################

class HapticMap:
    """Class wrapper for series of intensity mappings. Each mapping converts haptic intensity array
    into arrays of frequencies/periods and duty cycles, which can be sent to MINI switches via USB.
    
    NOTE: scaling factor of 2 is present due to microcontroller frequency of 2 kHz. If microcontroller
    firmware is updated (Jonathan), you must verify this scaling factor. Only period is scaled; duty cycle is 
    given in %."""

    scaling_factor = 2

    def linear_map(self, intensity_array: np.ndarray = np.zeros((4,7)), freq_range:tuple = (0,24), duty_range:tuple = (25,75)):
        """Converts intensity 0-1 to range specified in inputs via linear mapping.
        
        Inputs:
        -intensity_array: np.ndarray of shape (4,7) containing intensity of the frame. Default is zero array.
        -freq_range: tuple containing min and max frequency (Hz)
        -duty_range: tuple containing min and max duty cycle ratio (%)
        
        Using default values, intensity 0-1 will be mapped to 0-24 Hz, 25%-75%"""

        # map frequencies:
        mapped_freq = (freq_range[1]-freq_range[0])*intensity_array + freq_range[0]*np.ones(shape=intensity_array.shape) # linear mapped frequency (Hz)
        mapped_freq[mapped_freq==0] = 0.001 # can't be zero (div/0 when inverting to period)
        period_array = np.reciprocal(mapped_freq) # mapped period (sec)
        period_array = 1000*period_array # mapped period (ms)
        period_array = period_array.astype(int)
        period_array[period_array<np.floor(1000/freq_range[1])] = np.floor(1000/freq_range[1]) # threshold anything above freq limit
        period_array[period_array>500] = 0 # anything below 2 Hz = 0

        # map duty cycles:
        duty_array = (duty_range[1]-duty_range[0])*intensity_array + duty_range[0]*np.ones(shape=intensity_array.shape) # linear mapped duty (%)
        
        period_array = self.scaling_factor*period_array # scaling factor (see function description)
        output = {"duty": duty_array, "period": period_array}
        return output

############### INPUT FUNCTIONS + CLASS ################
class IntensityGenerator:
    """Class wrapper for functions that generate sequences of intensity arrays.
    
    Use this to create pre-programmed output sequences for the haptic display."""

    def __init__(self, total_time=3, frame_rate=24):
        """Inputs:
        -total_time: total time (seconds) of output to generate
        -frame_rate: frame rate for generated output"""
        self.t = np.arange(start=0, stop=total_time, step=1/frame_rate) # build time vector

    def make_output(self, output_array: np.ndarray):
        """Normalizes and list-izes the output sequence.
        
        Inputs:
        -output_array: np.ndarray of shape (4,7,n) for n total number of frames in the sequence.
        
        The output is normalize based on the maximum value in the entire output_array. This ensures the
        returned list never exceeds a value of 1.
        
        TODO: Also need to double check negatives/below zero values."""

        max_val = np.max(output_array)
        output = output_array/max_val # normalize to maximum value
        return list(np.unstack(output, axis=2)) # turn numpy array into a list of arrays

    def sawtooth(self, direction='left',scale=1, freq=1):
        """Sawtooth output signal.
        
        Inputs:
        -direction: direction of sawtooth on display. Options: 'left', 'right', 'up', 'down'
        -scale: the time scaling factor, which affects the width of the sawtooth
        -freq: frequency (Hz) of the sawtooth"""

        output = np.zeros((4,7,self.t.size))
        if direction=='left':
            for c in range(7):
                output[:,c,:] = 0.5 + 0.5*signal.sawtooth(freq*2*np.pi*(self.t+scale*c))
        elif direction=='right':
            for c in range(7):
                output[:,c,:] = 0.5 + 0.5*signal.sawtooth(-freq*2*np.pi*(self.t+scale*c))
        elif direction=='up':
            for r in range(4):
                output[r,:,:] = 0.5 + 0.5*signal.sawtooth(freq*2*np.pi*(self.t+scale*r))
        elif direction=='down':
            for r in range(4):
                output[r,:,:] = 0.5 + 0.5*signal.sawtooth(-freq*2*np.pi*(self.t+scale*r))
        return self.make_output(output)

    def sawtooth_global(self, freq=1):
        """Sawtooth global output signal (same for all taxels).
        
        Inputs:
        -freq: frequency (Hz) of the sawtooth"""

        output = np.zeros((4,7,self.t.size))
        ## I need to improve this part, remove for loops
        for r in range(4):
            for c in range(7):
                output[r,c,:] = 0.5 + 0.5*signal.sawtooth(freq*2*np.pi*self.t)
        return self.make_output(output)

    def sine(self, direction='left',scale=1, freq=1):
        """Sine output signal.
        
        Inputs:
        -direction: direction of sine on display. Options: 'left', 'right', 'up', 'down'
        -scale: the time scaling factor, which affects the width of the sine wave
        -freq: frequency (Hz) of the sine"""

        output = np.zeros((4,7,self.t.size))
        if direction=='left':
            for c in range(7):
                output[:,c,:] = 0.5 + 0.5*np.sin(freq*2*np.pi*(self.t+scale*c))
        elif direction=='right':
            for c in range(7):
                output[:,c,:] = 0.5 + 0.5*np.sin(-freq*2*np.pi*(self.t+scale*c))
        elif direction=='up':
            for r in range(4):
                output[r,:,:] = 0.5 + 0.5*np.sin(freq*2*np.pi*(self.t+scale*r))
        elif direction=='down':
            for r in range(4):
                output[r,:,:] = 0.5 + 0.5*np.sin(-freq*2*np.pi*(self.t+scale*r))
        return self.make_output(output)

    def sine_global(self, freq=1):
        """Sine global output signal (same for all taxels).
        
        Inputs:
        -freq: frequency (Hz) of the sine wave"""

        output = np.zeros((4,7,self.t.size))
        ## I need to improve this part, remove for loops
        for r in range(4):
            for c in range(7):
                output[r,c,:] = 0.5 + 0.5*np.sin(freq*2*np.pi*self.t) # need to make sure never exceeds range 0-1
        return self.make_output(output)

    def checker_square(self, freq=1):
        """Checkerboard pattern which alternates via square wave.
        
        Inputs:
        -freq: frequency (Hz) of the square wave switching."""

        output = np.zeros((4,7,self.t.size))
        ## I need to improve this part, remove for loops
        for r in range(4):
            for c in range(7):
                if (r%2==0 and c%2==0) or (r%2==1 and c%2==1): #if both r, c are even or odd:
                    output[r,c,:] = 0.5 + 0.5*signal.square(freq*2*np.pi*self.t) # need to make sure never exceeds range 0-1
                else:
                    output[r,c,:] = 0.5 - 0.5*signal.square(freq*2*np.pi*self.t) # need to make sure never exceeds range 0-1
        return self.make_output(output)

    def checker_sine(self, freq=1):
        """Checkerboard pattern which alternates via sine wave.
        
        Inputs:
        -freq: frequency (Hz) of the sine wave switching."""

        output = np.zeros((4,7,self.t.size))
        ## I need to improve this part, remove for loops
        for r in range(4):
            for c in range(7):
                if (r%2==0 and c%2==0) or (r%2==1 and c%2==1): #if both r, c are even or odd:
                    output[r,c,:] = 0.5 + 0.5*np.sin(freq*2*np.pi*self.t) # need to make sure never exceeds range 0-1
                else:
                    output[r,c,:] = 0.5 - 0.5*np.sin(freq*2*np.pi*self.t) # need to make sure never exceeds range 0-1
        return self.make_output(output)
    
    def ramp(self, direction=1):
        """Linear ramp from 0 to 1.
        
        Inputs:
        -direction: 1 for increasing ramp, -1 for decreasing ramp"""

        output = np.zeros((4,7,self.t.size))
        for r in range(4):
            for c in range(7):
                output[r,c,:] = 0.5 + 0.5*signal.sawtooth(direction*(1/self.t[-1])*2*np.pi*self.t)
        return self.make_output(output)