import numpy as np
import serial

class USBWriter:
    def __init__(self, serial_ports, serial_active:bool=True):
        self.serial_active = serial_active
        self.serial_list = self.initialize_ports(serial_ports) #expects a list with each list object being a serial object

    def initialize_ports(self, serial_ports):
        serial_list = []
        if self.serial_active:
            for port in serial_ports:
                serial_list.append(serial.Serial(port, 9600, timeout=0, bytesize=serial.EIGHTBITS))
        self.serial_list = serial_list

    def HV_enable(self):
        if self.serial_active:
            for ser in self.serial_list:
                ser.write('E'.encode())

    def HV_disable(self):
        if self.serial_active:
            for ser in self.serial_list:
                ser.write('D'.encode())

    def close_serial(self):
        if self.serial_active:
            for ser in self.serial_list:
                ser.close()

    def write_to_USB(self, output = dict(duty=np.zeros((7,4)), period=np.zeros((7,4)))):
        ## Assumes a linear/reshape-based encoding starting from top-left of display:
        # duty_array and period_array expect shape is (7,4)
            ##### CURRENT SETUP:
            # C2|B5|A8|A1
            # C3|B6|A9|A2
            # C4|B7|A10|A3
            # C5|B8|B1|A4
            # C6|B9|B2|A5
            # C7|B10|B3|A6
            # C8|C1|B4|A7

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
                # Make packets:
                packet_duty, packet_period = self.make_packets(duties, periods)
                # Write to USB:
                ser.write(packet_duty)
                ser.write(packet_period)

    def make_packets(self, duties, periods):
        duties_abs = np.int32(np.floor(np.multiply(periods,duties))) # convert from % to msec
        packetlist = []
        packetlist.append(('P').encode()) # encode start of period array
        for duty in duties_abs:
            packetlist.append((duty.item()).to_bytes(2, byteorder='little')) # convert to 16bit
        packet_duty = b''.join(packetlist) # combine packetlist as bytes

        packetlist = []
        packetlist.append(('T').encode()) # encode start of period array
        for period in periods:
            packetlist.append((period.item()).to_bytes(2, byteorder='little')) # convert to 16bit, factor of 2 for period in MCU
        packet_period = b''.join(packetlist) # combine packetlist as bytes

        return packet_duty, packet_period

def map_intensity(intensity_array):
    # this particular mapping maps to the range of 0-24 Hz
    ### IMPORTANT: scaling factor of 2 for microcontroller running at 2kHz (if firmware is updated, check this scaling value)
    # duty cycle is %, does not need to be scaled

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
    
    period_array = 2*period_array # scaling factor
    output = {"duty": duty_array, "period": period_array}
    return output


