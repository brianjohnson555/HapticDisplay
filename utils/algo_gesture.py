import numpy as np
import mediapipe as mp

# Google AI gesture recognizer setup:
class Recognizer:
    def __init__(self, gesture_data):
        # from Google AI tutorial:
        self.GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
        self.BaseOptions = mp.tasks.BaseOptions
        self.GestureRecognizer = mp.tasks.vision.GestureRecognizer
        self.GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        # initialize recognizer:
        self.recognizer = self.initialize_gesture_recognizer(gesture_data)

    def initialize_gesture_recognizer(self, gesture_data): #expects Gesture class object
        options = self.GestureRecognizerOptions(base_options=self.BaseOptions(model_asset_path='utils\gesture_recognizer.task'), # path for gesture recognizer file
                                        running_mode=self.VisionRunningMode.LIVE_STREAM, 
                                        result_callback=gesture_data.gesture_callback_from_recognizer)
        
        return self.GestureRecognizer.create_from_options(options)

# Create class to track gesture data
class Gesture:
    count_dict = {'None': 0, 
                  'Closed_Fist': 0, 
                  'Open_Palm': 0, 
                  'Pointing_Up': 0, 
                  'Thumb_Down': 0,
                  'Thumb_Up': 0,
                  'Victory': 0,
                  'ILoveYou': 0,
                  'Total': 0}
    gesture_latest = 'None'
    gesture_active = 'None'
    max_count = 30 # maximum cycles to count before rest
    margin = 0.5 # percentage of total gestures to count as 'active' gesture
    output_list = []
    output_latest = np.zeros((7,4))

    def update(self):
        self.update_gesture_count()
        self.update_output()

    def get_latest_gesture(self, recognizer, frame_timestamp_ms, frame):
        # recognizer has type: Recognizer.GestureRecognizer
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        recognizer.recognize_async(mp_image, frame_timestamp_ms)
        # recognizer initiates callback function, so it is equivalent here to running self.gesture_callback_from_recognizer()

    def gesture_callback_from_recognizer(self, result, output_image: mp.Image, timestamp_ms: int): #I'm not sure why, but the 3 inputs are required to work properly (even if unused)
        # result has type: Recognizer.GestureRecognizerResult
        if result.gestures: # if gesture was detected (not empty)
            self.gesture_latest = str(result.gestures[0][0].category_name)
        else:
            self.gesture_latest = 'None'

    def update_active_gesture(self):
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

        self.initialize_active_gesture()

    def update_gesture_count(self):
        if not self.output_list: # only run if output list is empty
            self.count_dict[self.gesture_latest]+=1
            self.count_dict['Total']+=1

            if self.count_dict['Total']>=self.max_count:
                self.update_active_gesture()

    def update_output(self):
        if not self.output_list: #if output_list is empty, set output to zeros
            self.output_latest = np.zeros((7,4))
        else:
            self.output_latest = self.output_list.pop() # take output from latest value in list

    def initialize_active_gesture(self):
        if self.gesture_active=='None':
            self.output_list = []
        elif self.gesture_active=='Closed_Fist':
            self.output_list = []
        elif self.gesture_active=='Open_Palm':
            self.output_list = []
        elif self.gesture_active=='Pointing_Up':
            self.output_list = []
        elif self.gesture_active=='Thumb_Down':
            self.output_list = []
        elif self.gesture_active=='Thumb_Up':
            self.output_list = []
        elif self.gesture_active=='Victory':
            self.output_list = []
        elif self.gesture_active=='ILoveYou':
            self.output_list = []

############################# haptic actuation sequences ################################
def sequence_thumb_up():
    output_list = []
    return output_list
