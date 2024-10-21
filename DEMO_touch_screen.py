#!/usr/bin/env python3

"""This demo script uses the built-in touch screen for real time feedback via GUI."""

###### USER SETTINGS ######
SERIAL_ACTIVE = True # if False, just runs the algorithm without sending to HV switches
COM_A = "COM9" # port for MINI switches 1-10
COM_B = "COM15" # port for MINI switches 11-20
COM_C = "COM16" # port for MINI swiches 21-28
FPS = 5 # update rate (frames per second)
SCALE = 200 # resolution scale for the GUI (pixels)
MAX_FREQ = 100 # maximum actuation frequency (Hz)
PRINT_OUTPUT = False # prints intensity output to terminal if True

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
slider = tk.Scale(master=frame_slider, from_=0, to=MAX_FREQ, orient="horizontal", 
                  length=SCALE*4, tickinterval=25, sliderlength=SCALE, width=SCALE,
                  label= "Frequency", font= tk.font.BOLD)
# slider.place(relx=.5, rely=.5, anchor="center")
slider.grid(row=0,column=0)
# HV toggle button
HV_on_toggle = tk.BooleanVar()
buttonHV_kwargs = dict(master=frame_slider, width=SCALE, height=SCALE*2, image=pixel, 
                 compound="c", font=tk.font.BOLD, variable=HV_on_toggle, 
                 onvalue=True, offvalue=False)
buttonHV = tk.Checkbutton(text="HV", **buttonHV_kwargs)
buttonHV.grid(row=0,column=1)

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
button_toggle2= tk.Button(text="Checker", **button_toggle_kwargs)
button_toggle2.grid(row=1, column=0)

##### define button functions/actuator outputs
def button_wiggle(freq=50):
    output = generator.ones_sequence(total_time=0.6, frame_rate=FPS, scale=freq/MAX_FREQ)
    return output
def button_sine(max_freq=50):
    output = generator.sine(total_time=5, frame_rate=FPS, scale=0.4, freq=1, max=(max_freq/MAX_FREQ))
    return output
def button_saw(max_freq=50):
    output = generator.checker_square(total_time=5, frame_rate=FPS, freq=1)
    return output
def slider_output(scale: tk.Scale):
    freq = scale.get()
    output = generator.ones_sequence(total_time=5, frame_rate=FPS, scale=freq/MAX_FREQ)
    return output

##### bind buttons
button_output = {"1":[],"2":[],"3":[],"4":[],"5":[],"6":[],"7":[],"8":[],"toggle1":[],"toggle2":[],"slider":[]}

def handle_button1(output):
    global button_output
    button_output["1"] = button_wiggle()
def handle_button2(event):
    global button_output
    button_output["2"] = button_wiggle()
def handle_button3(event):
    global button_output
    button_output["3"] = button_wiggle()
def handle_button4(event):
    global button_output
    button_output["4"] = button_wiggle()
def handle_button5(event):
    global button_output
    button_output["5"] = button_wiggle()
def handle_button6(event):
    global button_output
    button_output["6"] = button_wiggle()
def handle_button7(event):
    global button_output
    button_output["7"] = button_wiggle()
def handle_button8(event):
    global button_output
    button_output["8"] = button_wiggle()

button1.bind("<Button>", handle_button1) # bind to Next Frame button
button2.bind("<Button>", handle_button2) # bind to Next Frame button
button3.bind("<Button>", handle_button3) # bind to Next Frame button
button4.bind("<Button>", handle_button4) # bind to Next Frame button
button5.bind("<Button>", handle_button5) # bind to Next Frame button
button6.bind("<Button>", handle_button6) # bind to Next Frame button
button7.bind("<Button>", handle_button7) # bind to Next Frame button
button8.bind("<Button>", handle_button8) # bind to Next Frame button

toggle_choice = "sine"
def handle_toggle1(event):
    global button_output
    global toggle_choice
    toggle_choice = "sine"
    button_output["toggle1"] = button_sine()
def handle_toggle2(event):
    global button_output
    global toggle_choice
    toggle_choice = "saw"
    button_output["toggle2"] = button_saw()

button_toggle1.bind("<Button>", handle_toggle1)
button_toggle2.bind("<Button>", handle_toggle2)

##### define main actuation loop and run in separate thread #####
WINDOW_OPEN = True
def tk_close():
    global WINDOW_OPEN
    WINDOW_OPEN = False
    window.destroy() # destroy window
window.protocol("WM_DELETE_WINDOW", tk_close) # when window is closed, run tk_close

def get_latest_output():
    global button_output
    global slider
    global toggle_choice
    button_output["slider"] = slider_output(slider)
    current_output = dict()
    output = generator.zeros()[0]

    for key in button_output:
        if not button_output[key]:
            current_output[key] = generator.zeros()[0]
        else:
            current_output[key] = button_output[key].pop()

    output[0,0] = np.maximum(output[0,0], current_output["1"][0,0])
    output[0,1] = np.maximum(output[0,1], current_output["2"][0,1])
    output[1,0] = np.maximum(output[1,0], current_output["3"][1,0])
    output[1,1] = np.maximum(output[1,1], current_output["4"][1,1])
    output[2,0] = np.maximum(output[2,0], current_output["5"][2,0])
    output[2,1] = np.maximum(output[2,1], current_output["6"][2,1])
    output[3,0] = np.maximum(output[3,0], current_output["7"][3,0])
    output[3,1] = np.maximum(output[3,1], current_output["8"][3,1])
    output[0:2,2:6] = np.maximum(output[0:2,2:6], current_output["slider"][0:1,2:6])
    if toggle_choice=="sine":
        output[2:4,3:7] = np.maximum(output[2:4,3:7], current_output["toggle1"][2:4,3:7])
    elif toggle_choice=="saw":
        output[2:4,3:7] = np.maximum(output[2:4,3:7], current_output["toggle2"][2:4,3:7])

    if PRINT_OUTPUT:
        print(output)
    return output

def run_in_thread():
    global FPS
    global WINDOW_OPEN
    global HV_on_toggle
    HV_on_actual = False
    while WINDOW_OPEN: # run until GUI window is closed
        t_start = time.time()
        HV_button = HV_on_toggle.get()
        # check HV button and turn on/off HV:
        if HV_button is True and HV_on_actual is False:
            # Enable HV!!!
            serial_writer.HV_enable()
            HV_on_actual = True
            if PRINT_OUTPUT:
                print("-----HV IS ON!-----")
        elif HV_button is False and HV_on_actual is True:
            serial_writer.HV_disable()
            HV_on_actual = False
            if PRINT_OUTPUT:
                print("-----HV IS OFF!-----")
        
        output_array = get_latest_output()

        if HV_on_actual is True:
            duty_array, period_array = haptic_map.linear_map_single(output_array,
                                                                    freq_range=(0, 200),
                                                                    duty_range=(0.5, 0.5))
            serial_writer.write_array_to_USB(duty_array, period_array)
        # maintain constant fps:
        time.sleep(max(1/FPS-(time.time()-t_start), 0))

thread = threading.Thread(target=run_in_thread, daemon=True)
thread.start()

##### run tkinter #####
window.mainloop()

# after window is closed:
# Disable HV!!!
serial_writer.HV_disable()
zero_output = haptic_map.make_output_data(generator.zeros())
zero_intensity, zero_packets = zero_output.pop()
serial_writer.write_packets_to_USB(zero_packets)
time.sleep(1)