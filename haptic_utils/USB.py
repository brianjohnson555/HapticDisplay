#!/usr/bin/env python3

"""Sets up ***SerialWriter*** class to communicate with HV MINI switch via USB ports. 
Also contains packing functions to convert period and duty cycle values into byte strings for USB transfer.
**TODO**: this can be adapted to send voltage commands directly to the switch circuits."""

import numpy as np
import serial

class SerialWriter:
    """*SerialWriter* class handles packing data and writing to USB serial to control MINI switches."""

    def __init__(self, serial_ports: list = [], serial_active:bool=False):
        """Initializes the input ports for serial connection.

        **Parameters** :

        >>>**serial_ports** : list containing the serial ports to connect to. Expected format is [A, B, C] with A,B,C being the 
        string that defines the serial port. If A/B/C is None, the port will be ignored.
        
        >>>**serial_active** : bool which allows *SerialWriter* to be enabled or disabled via user parameter in the main
        demo script without having to modify code elsewhere. If *False*, no HV will be enabled and no data sent to USB"""
        
        self.serial_active = serial_active # SerialWriter will only write to USB if serial_active = True
        self.serial_list = []
        self.initialize_ports(serial_ports) #expects a list with each list object being a serial object.

    def initialize_ports(self, serial_ports: list):
        """@private Connects to USB serial ports. This is automatically called; you don't need to use this.
        
        **Parameters** :
        
        >>>**serial_ports** : *list* of serial ports imported from *__init__()*"""

        if self.serial_active:
            for port in serial_ports:
                if port: # if port is not None
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
        """Closes serial connection. 
        NOTE: this is currently redundant/not needed to run, because the serial connection closes
        automatically when a .py script finishes executing."""
        if self.serial_active:
            for ser in self.serial_list:
                ser.close()

    def write_array_to_USB(self, duty_array, period_array):
        """Writes the duty cycle and period data to USB. The write process is done in the order
         given by serial_list, which assumes the order is [MINI rack A, MINI rack B, MINI rack C].
        
        **Parameters** :
        
        >>>**duty_array** : np.ndarray of shape (30,) containing duty cycles [%]
        
        >>>**period_array** : np.ndarray of shape (30,) containing period values [ms]
        """

        ind = 0
        if self.serial_active:
            for ser in self.serial_list:
                # Parse duty cycles and periods from input:
                duties = duty_array[ind:ind+10]
                periods = period_array[ind:ind+10]
                ind=+10
                # Convert duties and periods into USB-transferable packets:
                packet = make_packet(duties, periods)
                # Write to USB:
                ser.write(packet)

    def write_packets_to_USB(self, packet_list):
        """Writes the duty cycle and period data to USB. The mapping from the (4,7) array
         to the USB is explained in README.md of this repository.
        
        **Parameters** :

        >>>**packet** : list containing three byte string elements corresponding to *COM_A*, *COM_B*, *COM_C*
        """

        if self.serial_active:
            for ser, packet in zip(self.serial_list, packet_list):
                ser.write(packet)
                

def make_packet(duties: np.ndarray, periods: np.ndarray):
    """Converts the duty cycle and period arrays into a USB-transferable packet.
    
    **Parameters** :

    >>>**duties** : np.ndarray of shape (10,) containing the duty cycle [%] of each MINI switch
    
    >>>**periods** : np.ndarray of shape (10,) containing the total period [ms] of each MINI switch
    
    **Returns** :
    >>>**packet** : combined byte string containing both period and total time [ms]"""

    duties_abs = np.int32(np.floor(np.multiply(periods,duties))) # convert from % to msec
    packetlist = []
    
    # encode start of period array
    packetlist.append(('P').encode()) 
    for duty in duties_abs:
        packetlist.append((duty.item()).to_bytes(2, byteorder='little')) # convert to 16bit

    # encode start of total array
    packetlist.append(('T').encode()) 
    for period in periods:
        packetlist.append((period.item()).to_bytes(2, byteorder='little')) # convert to 16bit
    packet = b''.join(packetlist) # combine packetlist as bytes

    return packet

def make_packet_list(duty_array: np.ndarray = np.zeros((30,),dtype=int), period_array: np.ndarray = np.zeros((30,),dtype=int)):
    """Converts the duty cycle and period arrays into a list of USB-transferable packets.
    This assumes mapping from (30,) array to 3 serial ports as explained in README.md of this repository.
    
    **Parameters** :

    >>>**duties** : np.ndarray of shape (30,) containing the duty cycle [%] of each MINI switch
    
    >>>**periods** : np.ndarray of shape (30,) containing the total period [ms] of each MINI switch
    
    **Returns** :
    >>>**packet_list** : list of combined byte strings containing both period and total time [ms]
    """
    
    packet_list = []
    #iterate for each serial port:
    for ii in [0, 10, 20]:
        # Parse duty cycles and periods from input:
        duties = duty_array[ii:ii+10]
        periods = period_array[ii:ii+10]

        duties_abs = np.int32(np.floor(np.multiply(periods,duties))) # convert from % to msec
        packetlist = []
        packetlist.append(('P').encode()) # encode start of period array
        for duty in duties_abs:
            packetlist.append((duty.item()).to_bytes(2, byteorder='little')) # convert to 16bit

        packetlist.append(('T').encode()) # encode start of total array
        for period in periods:
            packetlist.append((period.item()).to_bytes(2, byteorder='little')) # convert to 16bit
        packet = b''.join(packetlist) # combine packetlist as bytes

        # append to output:
        packet_list.append(packet)

    return packet_list

def make_packet_sequence(duty_array_list:list, period_array_list:list):
    """Converts lists of duty cycle and period information into a list of USB packets.
    Intention is to pre-process all packets before running real-time codes.
    
    **Parameters** :

    >>>**duty_array_list** : list of duty cycle arrays. Each list element is a flat np.ndarray of shape (30,)
    
    >>>**period_array_list** : list of period arrays. Each list element is a flat np.ndarray of shape (30,)
    
    **Returns** :
    
    >>>**packet_sequence** : list of packet_lists. Each element of *packet_sequence* is a list with
     three elements *[packet_A, packet_B, packet_C]* with A/B/C corresponding to USB serial output."""
    packet_sequence = []

    for duty_array, period_array in zip(duty_array_list, period_array_list):
        # convert to packets:
        packet_list = make_packet_list(duty_array, period_array)
        # append:
        packet_sequence.append(packet_list)

    return packet_sequence