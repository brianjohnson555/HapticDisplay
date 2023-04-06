'''
Author: Brian Johnson (bjohnson@is.mpg.de), 04.05.23
Created from 'Array3By4_Demo.py' by Yitian Shao (ytshao@is.mpg.de)
'''
import time
import numpy as np
import matplotlib.pyplot as plt
import nidaqmx as nidaq
import sys

# MARCO
F_CLK = 100000.0 # DOI Clock frequency (Circuit control sampling frequency, Hz)

NODE_NUM = 12
Channel_Num = 10

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

DIOout = np.empty((0,Channel_Num), dtype=np.uint8)

frameChargeRepNum =720 # Number of repetitions of charge per animation frame (per node) = 3600 (*NODE_NUM/F_CLK sec)
frameDischargeRepNum = 800 # Number of repetitions of discharge per animation frame (per node) = 4000  (*NODE_NUM/F_CLK sec)
nodeSeq = [0, 1, 2, 3, 7, 11, 10, 9, 8, 4]
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
    # DIOout = np.append(DIOout, np.zeros((int(frameIntvTime*F_CLK),10), dtype=np.uint8), axis=0)

DIOoutLen = DIOout.shape[0] # (= MeasureTime * F_PWM * int(F_CLK/F_PWM))
measureTime = (DIOoutLen/F_CLK)

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