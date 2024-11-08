#!/usr/bin/env python3

"""Functions to create pre-programmed intensity sequences for the haptic display."""

import numpy as np
from scipy import signal

def make_output(output_array: np.ndarray, normalize=True):
    """Normalizes and list-izes the output sequence. You shouldn't need to call this when running demos,
    it is only used internally for the generator functions. The output list is reversed, 
    so that .pop() draws the most recent output.

    The output is normalize based on the maximum value in the entire output_array. This ensures the
    returned list never contains a value >1.

    *TODO*: Also need to double check negatives/below zero values.

    **Parameters** :

    >>>**output_array** : np.ndarray of shape (4,7,n) for n total number of frames in the sequence.
    
    **Returns** : 

    >>>**output_list** : list of np arrays of shape (4,7), length of list==n

    """

    max_val = np.max(output_array)
    if max_val>0 and normalize is True:
        output = output_array/max_val # normalize to maximum value
    else:
        output = output_array
    output_list = list(np.unstack(output, axis=2))
    output_list.reverse()
    return output_list # turn numpy array into a list of arrays

def zeros_sequence(total_time:float=3, frame_rate:int=24):
    """Zero output signal of length input t. Duty cycle and period will be zero!!!
    
    **Parameters** :

    >>>**total_time** : total length of sequence [s]

    >>>**frame_rate** : speed/update rate [fps or Hz]
    
    **Returns** : 
    
    >>>**output_list** : output sequence of zeros, shape (4,7)"""

    t = np.arange(start=0, stop=total_time, step=1/frame_rate)
    output = np.zeros((4,7,t.size))
    return make_output(output)

def zeros():
    """Zero output signal, single [array]. Duty cycle and period will be zero!!!
    
    **Returns** : 
    
    >>>**output** : list which contains a zero np array, shape (4,7)"""

    output = np.zeros((4,7))
    return [output]

def ones_sequence(total_time:float=3, frame_rate:int=24, scale:float=1):
    """Uniform ones output signal of length input t. 'scale' changes the output magnitude.
    
    **Parameters** :

    >>>**total_time** : total length of sequence [s]

    >>>**frame_rate** : speed/update rate [fps or Hz]
    
    **Returns** : 
    
    >>>**output_list** : output sequence which is all ones, shape (4,7)"""

    t = np.arange(start=0, stop=total_time, step=1/frame_rate)
    output = scale*np.ones((4,7,t.size))
    return make_output(output, normalize=False)

def sawtooth(total_time:float=3, frame_rate:int=24, direction='left',scale:float=1, freq:float=1, max:float=1):
    """Sawtooth output signal.
    
    **Parameters** : 
    
    >>>**total_time** : total length of sequence [s]

    >>>**frame_rate** : speed/update rate [fps or Hz]

    >>>**direction** : direction of sawtooth on display. Options: 'left', 'right', 'up', 'down'

    >>>**scale** : the time scaling factor, which affects the width of the sawtooth
    
    >>>**freq** : frequency [Hz] of the sawtooth
    
    **Returns** : 
    
    >>>**output_list** : output sequence, as a list of arrays with shape (4,7)"""

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
    return make_output(max*output)

def sawtooth_global(total_time:float=3, frame_rate:int=24, freq:float=1):
    """Sawtooth global output signal (same for all taxels).
    
    **Parameters** :

    >>>**total_time** : total length of sequence [s]

    >>>**frame_rate** : speed/update rate [fps or Hz]

    >>>**freq** : frequency [Hz] of the sawtooth
    
    **Returns** : 
    
    >>>**output_list** : output sequence, as a list of arrays with shape (4,7)"""

    t = np.arange(start=0, stop=total_time, step=1/frame_rate)
    output = np.zeros((4,7,t.size))
    ## I need to improve this part, remove for loops
    for r in range(4):
        for c in range(7):
            output[r,c,:] = 0.5 + 0.5*signal.sawtooth(freq*2*np.pi*t)
    return make_output(output)

def sine(total_time:float=3, frame_rate:int=24, direction='left',scale:float=1, freq:float=1, max:float=1):
    """Sine output signal.
    
    **Parameters** :

    >>>**total_time** : total length of sequence [s]

    >>>**frame_rate** : speed/update rate [fps or Hz]

    >>>**direction** : direction of sine on display. Options: 'left', 'right', 'up', 'down'
    
    >>>**scale** : the time scaling factor, which affects the width of the sine wave
    
    >>>**freq** : frequency [Hz] of the sine
    
    **Returns** : 
    
    >>>**output_list** : output sequence, as a list of arrays with shape (4,7)"""

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
    return make_output(max*output)

def sine_global(total_time:float=3, frame_rate:int=24, freq:float=1):
    """Sine global output signal (same for all taxels).
    
    **Parameters** :
    
    >>>**total_time** : total length of sequence [s]
    
    >>>**frame_rate** : speed/update rate [fps or Hz]
    
    >>>**freq** : frequency [Hz] of the sine wave
    
    **Returns** : 
    
    >>>**output_list** : output sequence, as a list of arrays with shape (4,7)"""

    t = np.arange(start=0, stop=total_time, step=1/frame_rate)
    output = np.zeros((4,7,t.size))
    ## I need to improve this part, remove for loops
    for r in range(4):
        for c in range(7):
            output[r,c,:] = 0.5 + 0.5*np.sin(freq*2*np.pi*t) # need to make sure never exceeds range 0-1
    return make_output(output)

def checker_square(total_time:float=3, frame_rate:int=24, freq:float=1):
    """Checkerboard pattern which alternates via square wave.
    
    **Parameters** :
    
    >>>**total_time** : total length of sequence [s]
    
    >>>**frame_rate** : speed/update rate [fps or Hz]
    
    >>>**freq** : frequency [Hz] of the square wave switching.
    
    **Returns** : 
    
    >>>**output_list** : output sequence, as a list of arrays with shape (4,7)"""

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

def checker_sine(total_time:float=3, frame_rate:int=24, freq:float=1):
    """Checkerboard pattern which alternates via sine wave.
    
    **Parameters** :
    
    >>>**total_time** : total length of sequence [s]
    
    >>>**frame_rate** : speed/update rate [fps or Hz]
    
    >>>**freq** : frequency [Hz] of the sine wave switching.
    
    **Returns** : 
    
    >>>**output_list** : output sequence, as a list of arrays with shape (4,7)"""

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

def ramp(total_time:float=3, frame_rate:int=24, direction:int=1):
    """Linear ramp from 0 to 1.
    
    **Parameters** :
    
    >>>**total_time** : total length of sequence [s]
    
    >>>**frame_rate** : speed/update rate [fps or Hz]
    
    >>>**direction** : 1 for increasing ramp, -1 for decreasing ramp (1 to 0)
    
    **Returns** : 
    
    >>>**output_list** : output sequence, as a list of arrays with shape (4,7)"""

    t = np.arange(start=0, stop=total_time, step=1/frame_rate)
    output = np.zeros((4,7,t.size))
    for r in range(4):
        for c in range(7):
            output[r,c,:] = np.concatenate((0.5 + 0.5*direction*signal.sawtooth((1/t[-1])*2*np.pi*t[:-1]),[1]))
    return make_output(output)