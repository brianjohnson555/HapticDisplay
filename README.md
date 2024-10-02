# Haptic Display Project Codes

This repository is for all python-related codes for the transparent haptic display project.

Code authors: Brian K. Johnson

## File directory

The main directory contains the following file directory structure:

- **hapticdisplay/** (demo codes)
    - **algo_input_data/** (haptic data output from algorithm, for use in demos)
    - **algo_input_videos/** (videos for both algorithm and demos)
    - **figure_generation/** (code to generate plots, images for manuscript)
    - **output_images/** (any images generated for manuscript, presentations, etc.)
    - **output_videos/** (any videos generated for manuscript, presentations, etc.)
    - **preprocessing/** (visual-haptic algorithm code. Video goes in->haptic data comes out)
    - **haptic_utils/** (codebase of functions and classes to run the demos)


## Installation and requirements

The code is compiled in Python 3.10. Some files assume that the computer running the code has a built-in webcam.

The repository directory contains the file **`requirements.txt`** which lists all required Python packages. The necessary packages can be automatically installed by running `pip install -r requirements.txt` from the main directory.


## How to use this repository

After installing all necessary packages, the files can be used for a few different purposes:

#### 1. Running demos
For running demos of the haptic display, the main directory contains all demo codes. Each file is a separate demo (e.g. `DEMO_gesture.py` runs the haptic gesture detection demo). All required videos and haptic sequences for the demos should be included in the **`algo_input_data/`** and **`algo_input_videos/`** file directories.

#### 2. Generating new visual-haptic data to make new demos
Any new videos which you want to show on the display should be placed in the **`algo_input_videos/`** directory. Then, in **`pre_processing/`** directory, run the appropriate processing code to generate the haptic data. The haptic data will then be stored in **`algo_input_data/`**

#### 3. Creating new manually-authored haptic sequences
Codes within **`haptic_utils/`** can be used to create open-loop (non-algorithm based) input sequences to the haptic display for use in demos (e.g. the gesture detection demo). To use the existing scripts to write new code, import them with `import haptic_utils.haptic_map` etc.

#### 4. Working with manuscript figures
The **`figure_generation/`** directory can be used to write code to generate figures for the manuscript, presentations, etc.


## Haptic display wire mapping, demo script settings, script terminology

Since the haptic display is a 7x4 array (28 pixels) and we use 3 MINI switch racks (30 switches), the following wiring scheme is adopted to map from intensity array (np.ndarray) indices to the physical pixels:

| Screen top left |  |  | Landscape mode |  |  |  |
|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [0,0]=A1 | [0,1]=A2 | [0,2]=A3 | [0,3]=A4 | [0,4]=A5 | [0,5]=A6 | [0,6]=A7 |
| [1,0]=A8 | [1,1]=A9 | [1,2]=A10 | [1,3]=B1 | [1,4]=B2 | [1,5]=B3 | [1,6]=B4 |
| [2,0]=B5 | [2,1]=B6 | [2,2]=B7 | [2,3]=B8 | [2,4]=B9 | [2,5]=B10 | [2,6]=C1 |
| [3,0]=C2 | [3,1]=C3 | [3,2]=C4 | [3,3]=C5 | [3,4]=C6 | [3,5]=C7 | [3,6]=C8 |

A/B/C refers to the labeled MINI switch rack, and 1-10 refers to the MINI switch number (leftmost=1, rightmost=10). It's expected that when you initialize a USBWriter object with the serial ports,
the list of ports `[port_A, port_B, port_C]` will correspond to the MINI switch racks A/B/C.

#### Script terminology:
The following terms are adopted for naming variables/functions (my own personal conventions):

- `array`= Numpy np.ndarray object. E.g. the variable `duty_array` is a np.ndarray of duty cycle values.
- `list` = Python list object. E.g. the variable `duty_array_list` is a list of np.ndarrays of duty cycle values.
- `packet` = packaged data stream of bytes ready to be send to USB to control the MINI switches. E.g. the variable `packet_list` is a list of packets for each USB serial port. Send to USB via serial.write(packet).
- `intensity` = np.ndarray referring to 0-1 scale of intensity created from the visual-haptic algorithm. One intensity array of shape (4,7) is the output of the algorithm for a single
video frame, with the size (4,7) corresponding to the HASELs of the haptic display, and each element of the (4,7) array will range from 0 (least intense) to 1 (most intense). Through the haptic mapping, 
these intensities will be converted into periods and duty cycles for USB transfer.
- `sequence` = Python list which is a sequence of the haptic output/video frames. E.g. the variable `intensity_sequence` is a sequential list of intensity arrays, with the first item being the intensity of the first frame. 
Sequence implies time dependence. E.g. `packet_sequence` is a sequential list of packets to send to USB

## How to contribute/code conventions

All code is written with the following conventions, which should be maintained whereever new contributions are made to the code:

### Script headers
Each script starts with a docstring denoted by triple quotes `""" ... """` which describes
the basic purpose of the script. If the script is intended to be modified/worked with directly (demo scripts, etc.), then the following sections are also added using line comments (see **`DEMO_gesture.py`** for example):

```python
#!/usr/bin/env python3  <--- this makes sure the Python3 interpreter is used. Not strictly necessary for us.

"""Script description goes here."""

###### USER SETTINGS ######
USER_PARAM = True # if True, does x; if False, does y

###### INITIALIZATIONS ######
import numpy as np
import matplotlib.pyplot as plt
# etc.

###### MAIN ######
# rest of the code goes here
```

### Function/class headers
Like scripts, each new class, class method, or function must contain a docstring `""" ... """` that
descripts what the thing does. The docstring should also contain a list of required inputs. For example:

```python
class MyClass:
"""Description of class."""

def __init__(self, var1: int, var2: str):
    """Description of method.

    Inputs:
    -var1: int which does xyz
    -var2: str which does abc"""
```

### Script/variable/function/class names
Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#316-naming): `module_name`, `package_name`, `ClassName`, `method_name`, `function_name`, `local_var_name`, etc.

### User parameters
User parameters are defined in the `###### USER SETTINGS ######` header of the script.

- User variables should be upper case with underscores per word, like a global variable: `USER_VARIABLE = 5`.
- Define user variables with possible options as a comment: `SIGNAL_TYPE = 'sine' # sets type of output signal. Options: 'sine', 'square', or 'sawtooth'`

## License
The code is licensed under the MIT license viewable in **`LICENSE.txt`**