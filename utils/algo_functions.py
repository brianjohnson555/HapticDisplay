#!/usr/bin/env python3

"""Sets up USBWriter class to communicate with HV MINI switch USB, and preprogrammed outputs."""

import numpy as np
import serial
from scipy import signal

class USBWriter:
    """USBWriter class handles packing data and writing to USB serial to control MINI switches."""

    def __init__(self, serial_ports: list, serial_active:bool=True):
        """Initializes the input ports for serial connection.

        Inputs:
        -serial_ports: list containing the serial ports to connect to. Expected format is [A, B, C]
        -serial_active: bool which allows USBWriter to be enabled or disabled via user parameter in the main
        demo script without having to modify code elsewhere. If False, no HV will be enabled and no data sent to USB"""
        self.serial_active = serial_active # USBWriter will only write to USB if serial_active = True
        self.serial_list = self.initialize_ports(serial_ports) #expects a list with each list object being a serial object

    def initialize_ports(self, serial_ports: list):
        """Connects to USB serial ports.
        
        Inputs:
        serial_ports: list of serial ports imported from __init__()"""
        serial_list = []
        if self.serial_active:
            for port in serial_ports:
                serial_list.append(serial.Serial(port, 9600, timeout=0, bytesize=serial.EIGHTBITS))
        self.serial_list = serial_list

    def HV_enable(self):
        """Enables HV on all MINI switches."""
        if self.serial_active:
            for ser in self.serial_list:
                ser.write('E'.encode())

    def HV_disable(self):
        """Disables HV on all MINI switches."""
        if self.serial_active:
            for ser in self.serial_list:
                ser.write('D'.encode())

    def close_serial(self):
        """Closes serial connection. NOTE: this is currently redundant/not needed to run."""
        if self.serial_active:
            for ser in self.serial_list:
                ser.close()

    def write_to_USB(self, output = dict(duty=np.zeros((4,7)), period=np.zeros((4,7)))):
        """Writes the duty cycle and period data to USB.
        
        Inputs:
        -output: dict with keys 'duty' and 'period', each one containing np.ndarray of shape (4,7)
        
        The mapping from the (4,7) array to the USB is explained in README.md of this repository.
        The write process is done in the order given by serial_list, which assumes the order is
        [MINI rack A, MINI rack B, MINI rack C]."""

        # reshape array to flat list, then append 0s to end for 30 switches
        duty_flat = np.append(np.reshape(output["duty"],(1,28)), [0, 0])
        period_flat = np.append(np.reshape(output["period"],(1,28)), [0, 0])
        ind = 0
        if self.serial_active:
            for ser in self.serial_list:
                # Parse duty cycles and periods from input:
                duties = duty_flat[ind:ind+10]
                periods = period_flat[ind:ind+10]
                ind=+10
                # Convert duties and periods into USB-transferable packets:
                packet_duty, packet_period = self.make_packets(duties, periods)
                # Write to USB:
                ser.write(packet_duty)
                ser.write(packet_period)

    def make_packets(self, duties: np.ndarray, periods: np.ndarray):
        """Converts the duty cycle and period arrays into USB-transferable packets.
        
        Inputs:
        -duties: np.ndarray of shape (10,) containing the % duty cycle of each MINI switch
        -periods: np.ndarray of shape (10,) containing the total period (ms) of each MINI switch"""

        duties_abs = np.int32(np.floor(np.multiply(periods,duties))) # convert from % to msec
        packetlist = []
        packetlist.append(('P').encode()) # encode start of period array
        for duty in duties_abs:
            packetlist.append((duty.item()).to_bytes(2, byteorder='little')) # convert to 16bit
        packet_duty = b''.join(packetlist) # combine packetlist as bytes

        packetlist = []
        packetlist.append(('T').encode()) # encode start of period array
        for period in periods:
            packetlist.append((period.item()).to_bytes(2, byteorder='little')) # convert to 16bit
        packet_period = b''.join(packetlist) # combine packetlist as bytes

        return packet_duty, packet_period
    
############### INTENSITY MAPPINGS + CLASS ################

class IntensityMap:
    """Class wrapper for series of intensity mappings. Each mapping converts haptic intensity array
    into arrays of frequencies/periods and duty cycles, which can be sent to MINI switches via USB.
    
    NOTE: scaling factor of 2 is present due to microcontroller frequency of 2 kHz. If microcontroller
    firmware is updated, you must verify this scaling factor. Only period is scaled; duty cycle is 
    given in %."""

    scaling_factor = 2

    def map_0_24Hz(self, intensity_array: np.ndarray = np.zeros((4,7))):
        """Converts intensity 0-1 to range of 0-24 Hz with duty 33-74%.
        
        Inputs:
        -intensity_array: np.ndarray of shape (4,7) containing intensity of the frame. Default is zero array."""

        # map frequencies:
        mapped_freq = 24*intensity_array # mapped frequency (Hz)
        mapped_freq[mapped_freq==0] = 0.001 # can't be zero (div/0 when inverting to period)
        period_array = np.reciprocal(mapped_freq) # mapped period (sec)
        period_array = 1000*period_array # mapped period (ms)
        period_array = period_array.astype(int)
        period_array[period_array<42] = 42 # anything below 42 ms -> 42 ms (above 24 Hz -> 24 Hz)
        period_array[period_array>500] = 0 # anything above 500 ms -> 0 (below 2 Hz = 0)

        # map duty cycles:
        duty_array = np.power(mapped_freq.astype(int),0.25)/3 # map duty cycle directly based on frequency
        # Max duty = (24Hz^0.25)/3 = 74%
        # Min duty = (1Hz^0.25)/3 = 33%
        
        period_array = self.scaling_factor*period_array # scaling factor (see function description)
        output = {"duty": duty_array, "period": period_array}
        return output
    
    def map_0_200Hz(self, intensity_array: np.ndarray = np.zeros((4,7))):
        """Converts intensity 0-1 to range of 0-200 Hz with duty 20-75%.
        
        Inputs:
        -intensity_array: np.ndarray of shape (4,7) containing intensity of the frame. Default is zero array."""

        # map frequencies:
        mapped_freq = 200*intensity_array # mapped frequency (Hz)
        mapped_freq[mapped_freq==0] = 0.001 # can't be zero (div/0 when inverting to period)
        period_array = np.reciprocal(mapped_freq) # mapped period (sec)
        period_array = 1000*period_array # mapped period (ms)
        period_array = period_array.astype(int)
        period_array[period_array<5] = 5 # anything below 5 ms -> 5 ms (above 200 Hz -> 200 Hz)
        period_array[period_array>500] = 0 # anything above 500 ms -> 0 (below 2 Hz = 0)

        # map duty cycles:
        duty_array = (mapped_freq.astype(int)**0.25)/5 # map duty cycle directly based on frequency
        # Max duty = (200Hz^0.25)/5 = 75%
        # Min duty = (1Hz^0.25)/5 = 20%
        
        period_array = self.scaling_factor*period_array # scaling factor (see function description)
        output = {"duty": duty_array, "period": period_array}
        return output
    
    def map_0_24Hz_constant_duty(self, intensity_array: np.ndarray = np.zeros((4,7))):
        """Converts intensity 0-1 to range of 0-24 Hz with constant duty of 50%.
        
        Inputs:
        -intensity_array: np.ndarray of shape (4,7) containing intensity of the frame. Default is zero array."""

        # map frequencies:
        mapped_freq = 24*intensity_array # mapped frequency (Hz)
        mapped_freq[mapped_freq==0] = 0.001 # can't be zero (div/0 when inverting to period)
        period_array = np.reciprocal(mapped_freq) # mapped period (sec)
        period_array = 1000*period_array # mapped period (ms)
        period_array = period_array.astype(int)
        period_array[period_array<42] = 42 # anything below 42 ms -> 42 ms (above 24 Hz -> 24 Hz)
        period_array[period_array>500] = 0 # anything above 500 ms -> 0 (below 2 Hz = 0)

        # map duty cycles:
        duty_array = 0.5*np.ones((4,7)) # constant 50%
        
        period_array = self.scaling_factor*period_array # scaling factor (see function description)
        output = {"duty": duty_array, "period": period_array}
        return output
    
    def map_duty_constant_10Hz(self, intensity_array: np.ndarray = np.zeros((4,7))):
        """Converts intensity 0-1 to a constant 10 Hz with duty of 0-75%.
        
        Inputs:
        -intensity_array: np.ndarray of shape (4,7) containing intensity of the frame. Default is zero array."""

        # map frequencies:
        period_array = 100*np.ones((4,7)) # mapped period, constant 10 Hz (ms)
        # map duty cycles:
        duty_array = 0.75*intensity_array # intensity 1->0.75, 0->0 (linear scaling)
        
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