#!/usr/bin/env python3

"""Sets up Recognizer and Gesture classes + methods for gesture-based demo.

The Recognizer is based on a Google AI gesture_recognizer pre-trained model. There are 8
possible gestures recognized by the model: 'none', 'closed fist', 'open palm', 'pointing 
up', 'thumb down', 'thumb up', 'victory' (v-sign), and 'I love you' (in ASL)."""

import numpy as np
import mediapipe as mp
import utils.algo_functions as algo_functions
model_asset_path='utils\gesture_recognizer.task'

# Google AI gesture recognizer setup:
class Recognizer:
    """Initiates a pre-trained gesture recognition model.
    
    Calling GestureRecognizerResult retrieves the detected gesture of the input image. This uses a 
    callback function which is defined in the Gesture class."""

    def __init__(self, gesture_data):
        """Gesture model initialization.
        
        Inputs:
        -gesture_data: Gesture class object"""

        # from Google AI tutorial:
        self.GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
        self.BaseOptions = mp.tasks.BaseOptions
        self.GestureRecognizer = mp.tasks.vision.GestureRecognizer
        self.GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        # initialize recognizer:
        self.recognizer = self.initialize_gesture_recognizer(gesture_data)

    def initialize_gesture_recognizer(self, gesture_data):
        """Initializes the recognizer model using gesture_data, a Gesture class object with callback."""

        options = self.GestureRecognizerOptions(base_options=self.BaseOptions(model_asset_path), # path for gesture recognizer file
                                        running_mode=self.VisionRunningMode.LIVE_STREAM, 
                                        result_callback=gesture_data.gesture_callback_from_recognizer)
        
        return self.GestureRecognizer.create_from_options(options)

# Create class to track gesture data
input_generator = algo_functions.IntensityGenerator(3, 24)
class Gesture:
    """Contains all current gesture information and controls outputs to the haptic display.
    
    The class method gesture_callback_from_recognizer() must be an input in the initialization
     of the Recognizer class. It is the callback function used to retrieve the detected 
     gesture.
     
    The general method flow is as follows:
     1. __init__() called upon initialize of Gesture object.
     2. When get_latest_gesture() is called using a Recognizer class object, the Recognizer 
     calls gesture_callback_from_recognizer() and the latest detected gesture is retrieved.
     3. This automatically calls update(), which calls update_output() and update_gesture_count().
      4a. update_output() pops the latest intensity array from output_list, or sends zero array if empty.
      4b. update_gesture_count() adds to the count_dict['Total'] value. When the value exceeds
       max_count, it calls update_active_gesture().
      5. update_active_gesture() determines which gesture is 'active' base on the margin variable.
       After determining gesture_active, it calls run_active_gesture().
      6. run_active_gesture() populates the output_list with the corresponding sequence of outputs
       specified in output_dict. The list will be popped next time update()/update_output() is called."""
    
    def __init__(self):
        self.count_dict = {'None': 0,
                           'Closed_Fist': 0, 
                           'Open_Palm': 0, 
                           'Pointing_Up': 0, 
                           'Thumb_Down': 0,
                           'Thumb_Up': 0,
                           'Victory': 0,
                           'ILoveYou': 0,
                           'Total': 0}
        self.output_dict = self.initialize_active_gesture() # dictionary with pre-allocated output sequences
        self.gesture_latest = 'None' # latest gesture detected by the Recognizer class
        self.gesture_active = 'None' # gesture which is currently being executed (for output_list)
        self.max_count = 30 # maximum cycles to count before obtaining gesture_active
        self.margin = 0.5 # percentage of total gestures to count as 'active' gesture
        self.frame_rate = 24 # frame rate of video stream
        self.output_list = [] # list of current haptic intensities
        self.output_latest = np.zeros((4,7)) # haptic intensity passed to the display (HV!)

    def update(self):
        """Updates the intensity array output to the haptic display and updates gesture count
         if no more outputs are available."""
        
        self.update_output()
        if len(self.output_list)==0: # only run if output list is empty
            self.update_gesture_count()

    def get_latest_gesture(self, recognizer, frame_timestamp_ms: int, frame: np.ndarray):
        """Calls the Recognizer class object in livestream mode to obtain detected gesture on image frame.
        
        Inputs:
        -recognizer: Recognizer.GestureRecognizer object
        -frame_timestamp_ms: timestamp in milliseconds corresponding to captured video frame
        -frame: numpy array of the video frame grabbed from OpenCV"""

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        recognizer.recognize_async(mp_image, frame_timestamp_ms)
        # recognizer initiates callback function, so it is equivalent here to running self.gesture_callback_from_recognizer()
        self.update() # update obtained gesture info!

    def gesture_callback_from_recognizer(self, result, output_image: mp.Image, timestamp_ms: int): #I'm not sure why, but the 3 inputs are required to work properly (even if unused)
        """Callback function for the Recognizer class object to retrieve gesture info.
        
        Inputs:
        -result: Recognizer.GestureRecognizerResult object
        -output_image: mediapipe (mp) Image object
        -timestamp_ms: timestamp in milliseconds from the associate frame/gesture.
        
        NOTE: This is based on Google AI callback function. Although output_image and timestamp_ms are 
        unused in this method, they seem to be necessary in order for the GestureRecognizer callback to
        function properly."""

        if result.gestures: # if gesture was detected (not empty)
            # result.gesture contains lots of info, we only need the name of the detected gesture:
            self.gesture_latest = str(result.gestures[0][0].category_name)
        else: # no gesture was detected
            self.gesture_latest = 'None'

    def update_active_gesture(self):
        """Sets the active gesture based on the gesture that exceeds margin value."""

        top_gesture = None
        top_gesture_count = 0

        for key, value in self.count_dict.items(): # iterate through each count, find largest
            if value>top_gesture_count and key!='Total':
                top_gesture = key
                top_gesture_count = value
            self.count_dict[key] = 0 # reset value
            
        if top_gesture!='None' and top_gesture_count/self.max_count>=self.margin:
            self.gesture_active = top_gesture # active gesture is that which is above margin
        else:
            self.gesture_active = 'None'

        self.run_active_gesture()

    def update_gesture_count(self):
        """Updates the total gesture count to determine when to check for active gestures.

        The purpose of this method and the update_active_gesture() method are to make sure the
        active gesture is one which is intentionally shown by the user in the video (i.e. ignores
        gesture detection errors or 'noise' in the detection method.)"""

        self.count_dict[self.gesture_latest]+=1
        self.count_dict['Total']+=1

        if self.count_dict['Total']>=self.max_count:
            self.update_active_gesture()

    def update_output(self):
        """Sets the intensity array output to be mapped into haptic feedback (HV!)"""

        if len(self.output_list)==0: #if output_list is empty, set output to zeros
            self.output_latest = np.zeros((4,7))
        else:
            self.output_latest = self.output_list.pop() # take output from latest value in list

    def initialize_active_gesture(self):
        """Pre-allocates the output intensity sequence for each active gesture. The sequence is
         generated by generator functions in util.algo_functions."""
        
        output_dict = {}
        output_dict['None'] = [np.zeros((4,7))]
        output_dict['Closed_Fist'] = input_generator.checker_square(freq=3)
        output_dict['Open_Palm'] = [np.zeros((4,7))]
        output_dict['Pointing_Up'] = [np.zeros((4,7))]
        output_dict['Thumb_Down'] = input_generator.sawtooth(direction='left',scale=0.1, freq=3)
        output_dict['Thumb_Up'] = input_generator.sawtooth(direction='right',scale=0.1, freq=3)
        output_dict['Victory'] = [np.zeros((4,7))]
        output_dict['ILoveYou'] = [np.zeros((4,7))]
        return output_dict

    def run_active_gesture(self):
        """Populates the output list with the pre-allocated output sequence for the given gesture."""

        self.output_list = list(self.output_dict[self.gesture_active])

