#!/usr/bin/env python3

"""Mapping functions. Also defines class OutputData which collects intensity and packet sequences."""

import numpy as np
import haptic_utils.USB as USB

def linear_map_single(intensity_array:np.ndarray, freq_range:tuple = (0,24), duty_range:tuple = (0.05,0.5)):
    """Converts intensity 0-1 to range specified in inputs via linear mapping.
    
    Inputs:
    -intensity_array: np.ndarray containing intensity of the frame.
    -freq_range: tuple containing min and max frequency (Hz)
    -duty_range: tuple containing min and max duty cycle ratio (%)

    Outputs:
    -duty_array_flat: np.ndarray flattened to single dimension. Each element is taxel duty cycle (%)
    -period_array_flat: np.ndarray flattened to single dimension. Each element is taxel period (ms)
    
    Using default values, intensity 0-1 will be mapped to 0-24 Hz and 25%-75%
    
    NOTE: scaling factor of 2 is present due to microcontroller frequency of 2 kHz. If microcontroller
    firmware is updated (Jonathan), you must verify this scaling factor. Only period is scaled; duty cycle is 
    given in %."""
    scaling_factor = 2

    # map frequencies:
    mapped_freq = (freq_range[1]-freq_range[0])*intensity_array + freq_range[0]*np.ones(shape=intensity_array.shape) # linear mapped frequency (Hz)
    mapped_freq[mapped_freq==0] = 0.001 # can't be zero (div/0 when inverting to period)
    period_array = np.reciprocal(mapped_freq) # mapped period (sec)
    period_array = 1000*period_array # mapped period (ms)
    period_array = period_array.astype(int)
    # period_array[period_array<np.floor(1000/freq_range[1])] = np.floor(1000/freq_range[1]) # threshold anything above freq limit
    period_array[period_array>500] = 0 # anything below 1 Hz = 0

    # map duty cycles:
    duty_array = (duty_range[1]-duty_range[0])*intensity_array + duty_range[0]*np.ones(shape=intensity_array.shape) # linear mapped duty (%)
    period_array = scaling_factor*period_array # scaling factor (see function description)

    # reshape array to 1D, then append 0s to end for 30 switches total
        #   index 0:9 correspond to USB serial 1/MINI rack A
        #   index 10:19 correspond to USB serial 2/MINI rack B
        #   index 20:29 correspond to USB serial 3/MINI rack C (only 8 active, last 2 are 0)
    duty_array_flat = np.append(np.reshape(duty_array,(1,28)), [0, 0])
    period_array_flat = np.append(np.reshape(period_array,(1,28)), [0, 0])

    return duty_array_flat, period_array_flat

def linear_map_sequence(intensity_sequence:list, freq_range:tuple = (0,24), duty_range:tuple = (0.05,0.5)):
    """Runs method linear_map_single() on a sequence of intensities.
    
    Inputs:
    -intensity_sequence: list of np.ndarrays containing intensity of each frame. 
    -freq_range: tuple containing min and max frequency for the mapping (Hz)
    -duty_range: tuple containing min and max duty cycle ratio for the mapping(%)

    Outputs:
    -duty_array_list: list of flattened arrays of duty cycle (%)
    -period_array_list: list of flattened arrays of period (ms)
    
    Using default values, intensity 0-1 will be mapped to 0-24 Hz, 25%-75%."""

    # map frequencies:
    period_array_list = []
    duty_array_list = []

    for intensity_array in intensity_sequence:
        # map
        duty_array, period_array = linear_map_single(intensity_array, 
                                                freq_range, 
                                                duty_range)
        # append
        duty_array_list.append(duty_array)
        period_array_list.append(period_array)

    return duty_array_list, period_array_list


def make_output_data(intensity_sequence:list, **kwargs):
    """Passes sequence of intensities through a linear mapping and returns OutputData object.
    
    Inputs:
    -intensity_sequence: sequence of intensity arrays, expected shape of each array np.ndarray((4,7))
    -**kwargs: keyword arguments for the linear_map_sequence() function (e.g. frequency and duty range)

    Outputs:
    - OutputData object with fields .intensity_sequence (same as input) and .packet_sequence.
     packet_sequence is a list of packet_list objects which can be sent to USB."""
    
    # perform linear map:
    duty_array_list, period_array_list = linear_map_sequence(intensity_sequence, **kwargs)
    # convert to sequence of packets:
    packets_sequence = USB.make_packet_sequence(duty_array_list, period_array_list)
    # build OutputData object:
    output_data = OutputData(intensity_sequence, packets_sequence)
    return output_data

class OutputData:
    """OutputData class collects all necessary items for running close-loop haptic sequences.
    
    The only fields are self.intensity_sequence, which is the sequence of haptic intensity arrays,
    and self.packet_sequence, which is the sequence of packets that can be sent to USB to create
    haptic sensations.

    Since both data items in OutputData are lists, common list methods are used such as pop() and copy().
    
    Both items can be retreive with the pop() class method until each list is exhausted.
    
    Multiple OutputData objects can be combined with the extend() class method.
    
    NOTE: it is expected that the last index value e.g. intensity_sequence[-1] is the most recent
    in time, in this way pop() method will continue to draw latest values."""

    def __init__(self, intensity_sequence:list, packet_sequence:list):
        """Initializes OutputData object.
        
        Inputs:
        -intensity_sequence: sequency of np.ndarrays of intensity (shape (4,7))
        -packet_sequence: sequence of packet_list objects generated from haptic_utils.USB.py
        functions make_packet_sequence or make_packet_list."""

        self.intensity_sequence = intensity_sequence
        self.packet_sequence = packet_sequence

    def pop(self):
        """Returns latest value of intensity_sequence and packet_sequence with list.pop().
        
        Like a normal list pop(), this removes the item from the list.
        
        Outputs:
        - intensity_array, packet_list: array and packet which can be plotted and sent to USB."""

        if len(self.intensity_sequence)>0:
            return self.intensity_sequence.pop(), self.packet_sequence.pop()
        else:
            return np.zeros((4,7)), USB.make_packet_list()
        
    def get(self):
        """Returns the full value of intensity_sequence and packet_sequence."""

        return self.intensity_sequence, self.packet_sequence
        
    def copy(self):
        """Returns a OutputData class object which is a copy of the original."""

        return OutputData(self.intensity_sequence, self.packet_sequence)
    
    def extend(self, other_output_data):
        """Extends the intensity_sequence and packet_sequence.
        
        Because it is expected that the time-sequence of intensities and packets is reversed
        with index [-1] being the latest value, extend appends the new data to the front of the 
        list rather than the end. This way, pop() method continues to draw the latest values in 
        the correct order."""

        temp_list = other_output_data.intensity_sequence.copy()
        temp_list.extend(self.intensity_sequence)
        self.intensity_sequence = temp_list

        temp_list2 = other_output_data.packet_sequence.copy()
        temp_list2.extend(self.packet_sequence)
        self.packet_sequence = temp_list2

    def length(self):
        """Returns current length (type: int) of both the intensity and packet sequences."""
        return len(self.intensity_sequence)