'''
Control the new bipolar HV board (electrode-array) via NI DAQ.
(Optional) Measurement via NI DAQ.
Author: Yitian Shao (ytshao@is.mpg.de)
Created on 2021.09.17 based on 'B3HV4GNDDAQ (Obsolete).py'
'''
import time
import numpy as np
import matplotlib.pyplot as plt
import PyDAQmx as nidaq
import sys

# MARCO
F_CLK = 100000.0 # DOI Clock frequency (Circuit control sampling frequency, Hz)

NODE_NUM = 12 # 3HV * 4GND array of nodes
Channel_Num = 10 # 3HV*2 + 4GND channels

CHARGE_XY = [[0,6],[0,7],[0,8],[0,9],
             [2,6],[2,7],[2,8],[2,9],
            [4,6],[4,7],[4,8],[4,9]]

DISCHARGE_XY = [[1,6],[1,7],[1,8],[1,9],
                [3,6],[3,7],[3,8],[3,9],
                [5,6],[5,7],[5,8],[5,9]]

'''-------------------------------------------------------------------------------'''
'''Functions'''
def Percent2PWM(CLKnum, pinCharge, pinDischarge, pinGND, percentage = 0.0):
    PWMratioInd = int(percentage * 0.01 * CLKnum)

    if(PWMratioInd < 0):
        PWMratioInd = 0

    PWMout = np.zeros((CLKnum, Channel_Num), dtype=np.uint8)

    PWMout[:PWMratioInd, pinCharge] = 1 # Charge cycle

    PWMout[(PWMratioInd+1):, pinDischarge] = 1 # Discharge cycle, must skip 1 tick for safety

    PWMout[:, pinGND] = 1  # GND

    return PWMout
'''-------------------------------------------------------------------------------'''

# Parameters
frameIntvTime = 0.01 # (sec) Time pause interval between two frames

usePWM = 0 # Turn on PWM control, otherwise perform constant charge/discharge action for each frame

DIOout = np.empty((0,Channel_Num), dtype=np.uint8)

if (usePWM):
    F_PWM = 1000  # PWM frequency (Output waveform sampling frequency, Hz)

    # PWM info
    PWMpulseWidth = 1.0 / F_PWM
    PWMpulseCLKnum = int(F_CLK * PWMpulseWidth)  # Total sample number per PWM segment
    print("PWM frequency = %.1f Hz, PWM duration = %.1f us (%d CLK ticks)"%(F_PWM, 1e6 * PWMpulseWidth, PWMpulseCLKnum))

    # Generate sinusoid signals
    sinDuration = 10.0  # Total time duration (sec)
    sinFreq = 250

    t = np.arange(int(sinDuration * F_PWM)) / F_PWM
    y = -50 * np.cos(2 * np.pi * sinFreq * t) + 50 # y ranged from 0 to 100

    actNode = 6 # Range from 0 to 11

    oneBlock = np.array([Percent2PWM(PWMpulseCLKnum, CHARGE_XY[actNode][0], DISCHARGE_XY[actNode][0],
                                     CHARGE_XY[actNode][1], yi) for yi in y], dtype=np.uint8)
    oneBlock = oneBlock.reshape((-1, Channel_Num))

    DIOout = np.append(DIOout, oneBlock, axis=0)

    # Discharge for 1.0 second
    for discharge_i in [1,3,5]:
        dischargeBlock = np.zeros((int(1.0*F_CLK), Channel_Num), dtype=np.uint8)
        dischargeBlock[:, [discharge_i,6,7,8,9]] = 1
        DIOout = np.append(DIOout, dischargeBlock, axis=0)

else: # DC activation signal for each frame
    frameChargeRepNum =1440 # Number of repetitions of charge per animation frame (per node) = 3600 (*NODE_NUM/F_CLK sec)
    frameDischargeRepNum = 1600 # Number of repetitions of discharge per animation frame (per node) = 4000  (*NODE_NUM/F_CLK sec)

    if 0:
        animation = np.array([
        [0, 0, 0, 0,
        0, 0, 0, 0,
        1, 0, 0, 0],
        [0, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 0, 0],

        [-1, -1, -1, -1,
        -1, -1, -1, -1,
        -1, -1, -1, -1],

        [0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 1],
        [0, 0, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 0],

        [-1, -1, -1, -1,
        -1, -1, -1, -1,
        -1, -1, -1, -1],

        [0, 0, 0, 1,
        0, 0, 0, 0,
        0, 0, 0, 0],
        [0, 0, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 0],

        [-1, -1, -1, -1,
        -1, -1, -1, -1,
        -1, -1, -1, -1],

        [1, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0],
        [0, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 0, 0],

        [-1, -1, -1, -1,
        -1, -1, -1, -1,
        -1, -1, -1, -1]
        ])

    # Going around ---------------------------------------------------
    else:
        nodeSeq = [0, 1, 2, 3, 7, 11, 10, 9, 8, 4]
        # nodeSeq = [4, 8, 9, 10, 11, 7, 3, 2, 1, 0]
        animation = np.empty((0,12))
        for nodeSi in nodeSeq:
            temp = np.zeros((1,12))
            temp[0,nodeSi] = 1
            animation = np.append(animation, temp, axis=0)
            animation = np.append(animation, -1*np.ones((1,12)), axis=0)
        print(animation)

    frameNum = animation.shape[0]

    # Constant charging (DC)
    dischargeNode = np.arange(start=0, stop=NODE_NUM * frameDischargeRepNum, step=NODE_NUM) # Discharge node discrete indices
    chargeNode = dischargeNode[:frameChargeRepNum]  # Charging node indices

    for animFrame in animation:
        # Each frame: Node number * Repetition of tick per node * (1 active tick + 1 empty tick)
        oneFrame = np.zeros((NODE_NUM*frameDischargeRepNum*2,Channel_Num), dtype=np.uint8)

        chargeInd = np.where(animFrame == 1)[0]
        for i in chargeInd:
            oneFrame[(chargeNode+i)*2, CHARGE_XY[i][0]] = 1
            oneFrame[(chargeNode+i)*2, CHARGE_XY[i][1]] = 1

        dischargeInd = np.where(animFrame == -1)[0]
        for i in dischargeInd:
            oneFrame[(dischargeNode+i)*2, DISCHARGE_XY[i][0]] = 1
            oneFrame[(dischargeNode+i)*2, DISCHARGE_XY[i][1]] = 1

        DIOout = np.append(DIOout, oneFrame, axis=0)

# Safety measure
DIOout[-1,:] = 0 # Ensure all channels are turned off at the end

DIOoutLen = DIOout.shape[0] # (= MeasureTime * F_PWM * int(F_CLK/F_PWM))
print("DIOout Shape: ", DIOout.shape)

measureTime = (DIOoutLen/F_CLK)
print("Total time = %.3f sec" % measureTime)
'''-------------------------------------------------------------------------------'''
''' Debug Tool'''
dispOutput = 1
if dispOutput:
    fig1 = plt.figure(figsize = (16,6))
    fig1.suptitle(("CLK Freq = %.0f Hz" % F_CLK), fontsize=12)
    ax = fig1.add_subplot(111)
    t = np.arange(DIOoutLen)/F_CLK
    for i in range(10):
        ax.plot(t, DIOout[:,i]*0.5+i, '-')
        # ax.plot(t, DIOout[:, i] + i * 2, '.-')
    plt.show()
'''-------------------------------------------------------------------------------'''
if __name__ == '__main__':
    with nidaq.Task() as task1:
        # DAQ configuration
        task1.CreateDOChan("Dev1/port0/line0:5,Dev1/port0/line10:13", None, nidaq.DAQmx_Val_ChanPerLine)
        task1.CfgSampClkTiming(source="OnboardClock", rate=F_CLK, activeEdge=nidaq.DAQmx_Val_Rising,
                               sampleMode=nidaq.DAQmx_Val_FiniteSamps, sampsPerChan=DIOoutLen)

        task1.WriteDigitalLines(numSampsPerChan=DIOoutLen, autoStart=False,
                                timeout=nidaq.DAQmx_Val_WaitInfinitely, dataLayout=nidaq.DAQmx_Val_GroupByScanNumber,
                                writeArray=DIOout, reserved=None, sampsPerChanWritten=None)

        # ------------ start ------------ #
        task1.StartTask()
        print("Start sampling...")

        time.sleep(measureTime + 0.1)

        task1.StopTask()
        print("Task completed!")
        task1.ClearTask()

'''-------------------------------------------------------------------------------'''