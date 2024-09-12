#!/usr/bin/env python3

"""Functions to create pre-programmed intensity sequences for the haptic display."""

import numpy as np
from scipy import signal

def make_output(output_array: np.ndarray):
    """Normalizes and list-izes the output sequence.
    
    Inputs:
    -output_array: np.ndarray of shape (4,7,n) for n total number of frames in the sequence.
    
    The output is normalize based on the maximum value in the entire output_array. This ensures the
    returned list never exceeds a value of 1.

    NOTE: The output list is reversed, so that .pop() draws the most recent output.
    
    TODO: Also need to double check negatives/below zero values."""

    max_val = np.max(output_array)
    output = output_array/max_val # normalize to maximum value
    output_list = list(np.unstack(output, axis=2))
    output_list.reverse()
    return output_list # turn numpy array into a list of arrays

def zeros_sequence(total_time=3, frame_rate=24):
    """Zero output signal of length input t. No duty cycle or period. All zeros!!!"""

    t = np.arange(start=0, stop=total_time, step=1/frame_rate)
    output = np.zeros((4,7,t.size))
    return make_output(output)

def zeros():
    """Zero output signal, single array. No duty cycle or period. All zeros!!!"""

    output = np.zeros((4,7))
    return [output]

def sawtooth(total_time=3, frame_rate=24, direction='left',scale=1, freq=1):
    """Sawtooth output signal.
    
    Inputs:
    -direction: direction of sawtooth on display. Options: 'left', 'right', 'up', 'down'
    -scale: the time scaling factor, which affects the width of the sawtooth
    -freq: frequency (Hz) of the sawtooth"""

    t = np.arange(start=0, stop=total_time, step=1/frame_rate)
    output = np.zeros((4,7,t.size))
    if direction=='left':
        for c in range(7):
            output[:,c,:] = 0.5 + 0.5*signal.sawtooth(freq*2*np.pi*(t+scale*c))
    elif direction=='right':
        for c in range(7):
            output[:,c,:] = 0.5 + 0.5*signal.sawtooth(-freq*2*np.pi*(t+scale*c))
    elif direction=='up':
        for r in range(4):
            output[r,:,:] = 0.5 + 0.5*signal.sawtooth(freq*2*np.pi*(t+scale*r))
    elif direction=='down':
        for r in range(4):
            output[r,:,:] = 0.5 + 0.5*signal.sawtooth(-freq*2*np.pi*(t+scale*r))
    return make_output(output)

def sawtooth_global(total_time=3, frame_rate=24, freq=1):
    """Sawtooth global output signal (same for all taxels).
    
    Inputs:
    -freq: frequency (Hz) of the sawtooth"""

    t = np.arange(start=0, stop=total_time, step=1/frame_rate)
    output = np.zeros((4,7,t.size))
    ## I need to improve this part, remove for loops
    for r in range(4):
        for c in range(7):
            output[r,c,:] = 0.5 + 0.5*signal.sawtooth(freq*2*np.pi*t)
    return make_output(output)

def sine(total_time=3, frame_rate=24, direction='left',scale=1, freq=1):
    """Sine output signal.
    
    Inputs:
    -direction: direction of sine on display. Options: 'left', 'right', 'up', 'down'
    -scale: the time scaling factor, which affects the width of the sine wave
    -freq: frequency (Hz) of the sine"""

    t = np.arange(start=0, stop=total_time, step=1/frame_rate)
    output = np.zeros((4,7,t.size))
    if direction=='left':
        for c in range(7):
            output[:,c,:] = 0.5 + 0.5*np.sin(freq*2*np.pi*(t+scale*c))
    elif direction=='right':
        for c in range(7):
            output[:,c,:] = 0.5 + 0.5*np.sin(-freq*2*np.pi*(t+scale*c))
    elif direction=='up':
        for r in range(4):
            output[r,:,:] = 0.5 + 0.5*np.sin(freq*2*np.pi*(t+scale*r))
    elif direction=='down':
        for r in range(4):
            output[r,:,:] = 0.5 + 0.5*np.sin(-freq*2*np.pi*(t+scale*r))
    return make_output(output)

def sine_global(total_time=3, frame_rate=24, freq=1):
    """Sine global output signal (same for all taxels).
    
    Inputs:
    -freq: frequency (Hz) of the sine wave"""

    t = np.arange(start=0, stop=total_time, step=1/frame_rate)
    output = np.zeros((4,7,t.size))
    ## I need to improve this part, remove for loops
    for r in range(4):
        for c in range(7):
            output[r,c,:] = 0.5 + 0.5*np.sin(freq*2*np.pi*t) # need to make sure never exceeds range 0-1
    return make_output(output)

def checker_square(total_time=3, frame_rate=24, freq=1):
    """Checkerboard pattern which alternates via square wave.
    
    Inputs:
    -freq: frequency (Hz) of the square wave switching."""

    t = np.arange(start=0, stop=total_time, step=1/frame_rate)
    output = np.zeros((4,7,t.size))
    ## I need to improve this part, remove for loops
    for r in range(4):
        for c in range(7):
            if (r%2==0 and c%2==0) or (r%2==1 and c%2==1): #if both r, c are even or odd:
                output[r,c,:] = 0.5 + 0.5*signal.square(freq*2*np.pi*t) # need to make sure never exceeds range 0-1
            else:
                output[r,c,:] = 0.5 - 0.5*signal.square(freq*2*np.pi*t) # need to make sure never exceeds range 0-1
    return make_output(output)

def checker_sine(total_time=3, frame_rate=24, freq=1):
    """Checkerboard pattern which alternates via sine wave.
    
    Inputs:
    -freq: frequency (Hz) of the sine wave switching."""

    t = np.arange(start=0, stop=total_time, step=1/frame_rate)
    output = np.zeros((4,7,t.size))
    ## I need to improve this part, remove for loops
    for r in range(4):
        for c in range(7):
            if (r%2==0 and c%2==0) or (r%2==1 and c%2==1): #if both r, c are even or odd:
                output[r,c,:] = 0.5 + 0.5*np.sin(freq*2*np.pi*t) # need to make sure never exceeds range 0-1
            else:
                output[r,c,:] = 0.5 - 0.5*np.sin(freq*2*np.pi*t) # need to make sure never exceeds range 0-1
    return make_output(output)

def ramp(total_time=3, frame_rate=24, direction=1):
    """Linear ramp from 0 to 1.
    
    Inputs:
    -direction: 1 for increasing ramp, -1 for decreasing ramp (1 to 0)"""

    t = np.arange(start=0, stop=total_time, step=1/frame_rate)
    output = np.zeros((4,7,t.size))
    for r in range(4):
        for c in range(7):
            output[r,c,:] = np.concatenate((0.5 + 0.5*direction*signal.sawtooth((1/t[-1])*2*np.pi*t[:-1]),[1]))
    return make_output(output)