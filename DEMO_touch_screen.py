#!/usr/bin/env python3

"""This demo script uses the built-in touch screen for real time feedback via GUI."""

###### USER SETTINGS ######
SERIAL_ACTIVE = False # if False, just runs the algorithm without sending to HV switches
COM_A = "COM9" # port for MINI switches 1-10
COM_B = "COM15" # port for MINI switches 11-20
COM_C = "COM16" # port for MINI swiches 21-28
FPS = 10 # update rate (frames per second)
SCALE = 200 # resolution scale for the GUI

###### INITIALIZATIONS ######
import time
import haptic_utils.haptic_map as haptic_map
import haptic_utils.generator as generator
import haptic_utils.USB as USB
import tkinter as tk
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #interface matplotlib with tkinter
import numpy as np

###### MAIN ######

##### Set up USBWriter:
serial_ports = [COM_A, COM_B, COM_C]
serial_writer = USB.SerialWriter(serial_ports, serial_active=SERIAL_ACTIVE)
time.sleep(1)

##### Setting up Tkinter
window = tk.Tk()
pixel = tk.PhotoImage(width=1, height=1)

##### set up big windows
frame_left = tk.Frame(master=window)
frame_left.grid(row=0, column=0)
frame_right = tk.Frame(master=window)
frame_right.grid(row=0, column=1)

##### set up inner windows
## button window:
frame_buttons = tk.Frame(master=frame_left, width=SCALE*2, height=SCALE*4, bg='red')
frame_buttons.pack()
button_kwargs = dict(master=frame_buttons, width=SCALE, height=SCALE, image=pixel, 
                 compound="c", font=tk.font.BOLD)
button1 = tk.Button(text="1", **button_kwargs)
button1.grid(row=0, column=0)
button2 = tk.Button(text="2", **button_kwargs)
button2.grid(row=0, column=1)
button3 = tk.Button(text="3", **button_kwargs)
button3.grid(row=1, column=0)
button4 = tk.Button(text="4", **button_kwargs)
button4.grid(row=1, column=1)
button5 = tk.Button(text="5", **button_kwargs)
button5.grid(row=2, column=0)
button6 = tk.Button(text="6", **button_kwargs)
button6.grid(row=2, column=1)
button7 = tk.Button(text="7", **button_kwargs)
button7.grid(row=3, column=0)
button8 = tk.Button(text="8", **button_kwargs)
button8.grid(row=3, column=1)

## slider window:
frame_slider = tk.Frame(master=frame_right, width=SCALE*5, height=SCALE*2)
frame_slider.grid(row=0, column=0)
slider = tk.Scale(master=frame_slider, from_=0, to=200, orient="horizontal", 
                  length=SCALE*5, tickinterval=25, sliderlength=SCALE, width=SCALE,
                  label= "Frequency", font= tk.font.BOLD)
slider.place(relx=.5, rely=.5, anchor="center")

## toggle window:
frame_toggle = tk.Frame(master=frame_right, width=SCALE*5, height=SCALE*2, bg='blue')
frame_toggle.grid(row=1, column=0)
frame_toggle_left = tk.Frame(master=frame_toggle, width=SCALE*2, height=SCALE*2)
frame_toggle_right = tk.Frame(master=frame_toggle, width=SCALE*4, height=SCALE*2, bg='blue')
frame_toggle_left.grid(row=0, column=0)
frame_toggle_right.grid(row=0, column=1)
button_toggle_kwargs = dict(master=frame_toggle_left, width=SCALE, height=SCALE, image=pixel, 
                 compound="c", font=tk.font.BOLD)
button_toggle1= tk.Button(text="Sine", **button_toggle_kwargs)
button_toggle1.grid(row=0, column=0)
button_toggle2= tk.Button(text="Sawtooth", **button_toggle_kwargs)
button_toggle2.grid(row=1, column=0)

##### define button functions/actuator outputs
def button_wiggle(row, col, freq=30):
    return
def scale_output(scale: tk.Scale):
    freq = scale.get()

##### bind buttons
def handle_button1(event):
    button_wiggle(0, 0)
def handle_button2(event):
    button_wiggle(0, 1)
def handle_button3(event):
    button_wiggle(1, 0)
def handle_button4(event):
    button_wiggle(1, 1)
def handle_button5(event):
    button_wiggle(2, 0)
def handle_button6(event):
    button_wiggle(2, 1)
def handle_button7(event):
    button_wiggle(3, 0)
def handle_button8(event):
    button_wiggle(3, 1)
button1.bind("<Button>", handle_button1) # bind to Next Frame button
button2.bind("<Button>", handle_button2) # bind to Next Frame button
button3.bind("<Button>", handle_button3) # bind to Next Frame button
button4.bind("<Button>", handle_button4) # bind to Next Frame button
button5.bind("<Button>", handle_button5) # bind to Next Frame button
button6.bind("<Button>", handle_button6) # bind to Next Frame button
button7.bind("<Button>", handle_button7) # bind to Next Frame button
button8.bind("<Button>", handle_button8) # bind to Next Frame button

def handle_toggle1(event):
    button_wiggle(2, 2)
def handle_toggle2(event):
    button_wiggle(3, 2)
button_toggle1.bind("<Button>", handle_toggle1)
button_toggle2.bind("<Button>", handle_toggle2)

# Enable HV!!!
serial_writer.HV_enable()
time.sleep(0.5)

##### define main actuation loop and run in separate thread #####
WINDOW_OPEN = True
def tk_close():
    global WINDOW_OPEN
    WINDOW_OPEN = False
    window.destroy() # destroy window
window.protocol("WM_DELETE_WINDOW", tk_close) # when window is closed, run tk_close

def run_in_thread():
    global FPS
    global WINDOW_OPEN
    while WINDOW_OPEN: # run until GUI window is closed
        t_start = time.time()
        # do things here
        # maintain constant fps:
        time.sleep(max(1/FPS-(time.time()-t_start), 0))

thread = threading.Thread(target=run_in_thread, daemon=True)
thread.start()

##### run tkinter #####
window.mainloop()

# Disable HV!!!
serial_writer.HV_disable()
zero_output = haptic_map.make_output_data(generator.zeros())
zero_intensity, zero_packets = zero_output.pop()
serial_writer.write_packets_to_USB(zero_packets)
time.sleep(1)