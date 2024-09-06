#!/usr/bin/env python3

"""Sets up USBWriter class to communicate with HV MINI switch USB"""

import numpy as np
import serial

class USBWriter:
    """USBWriter class handles packing data and writing to USB serial to control MINI switches."""

    def __init__(self, serial_ports: list = [], serial_active:bool=False):
        """Initializes the input ports for serial connection.

        Inputs:
        -serial_ports: list containing the serial ports to connect to. Expected format is [A, B, C]
        -serial_active: bool which allows USBWriter to be enabled or disabled via user parameter in the main
        demo script without having to modify code elsewhere. If False, no HV will be enabled and no data sent to USB"""
        self.serial_active = serial_active # USBWriter will only write to USB if serial_active = True
        self.serial_list = []
        self.initialize_ports(serial_ports) #expects a list with each list object being a serial object

    def initialize_ports(self, serial_ports: list):
        """Connects to USB serial ports.
        
        Inputs:
        serial_ports: list of serial ports imported from __init__()"""

        if self.serial_active:
            for port in serial_ports:
                self.serial_list.append(serial.Serial(port, 9600, timeout=0, bytesize=serial.EIGHTBITS))

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

    def write_to_USB(self, duty_array, period_array):
        """Writes the duty cycle and period data to USB.
        
        Inputs:
        -output: dict with keys 'duty' and 'period', each one containing np.ndarray of shape (4,7)
        
        The mapping from the (4,7) array to the USB is explained in README.md of this repository.
        The write process is done in the order given by serial_list, which assumes the order is
        [MINI rack A, MINI rack B, MINI rack C]."""

        # reshape array to flat list, then append 0s to end for 30 switches
        duty_flat = np.append(np.reshape(duty_array,(1,28)), [0, 0])
        period_flat = np.append(np.reshape(period_array,(1,28)), [0, 0])
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