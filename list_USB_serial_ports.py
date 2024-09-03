#!/usr/bin/env python3

"""Run this script to see which USB ports are being detected.

This is useful to determine which port corresponds to which MINI switch module."""

import serial.tools.list_ports
ports = serial.tools.list_ports.comports()

for port, desc, hwid in sorted(ports):
        print("{}: {} [{}]".format(port, desc, hwid)) # print data associated with each port